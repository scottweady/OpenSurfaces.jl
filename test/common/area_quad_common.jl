# Shared by test_quadrature.jl (regression) and validation/val_area_quad.jl (convergence
# sweeps): scalar test kernels, the exact reference (divergence theorem), test panels and targets, a
# disk mesh, the dense smooth rule, and a grid mesh for timing.
using OpenSurfaces, LinearAlgebra, StaticArrays, SparseArrays, Test, QuadGK, Printf
using OpenSurfaces: gausslegendre

# four scalar kernels packed in a 2×2 matrix: [log ρ, 1/ρ, r₁r₂/ρ², r₁r₂/ρ³] (column-major)
Kscal(r) = (ρ = norm(r); ρ == 0 ? zero(SMatrix{2,2,Float64,4}) :
            SMatrix{2,2,Float64,4}(log(ρ), 1 / ρ, r[1] * r[2] / ρ^2, r[1] * r[2] / ρ^3))
packed(M) = SVector{4,Float64}(M[1], M[2], M[3], M[4])

G3k(r) = G3(r; μ_3D=2.0)             # rigid-disc value 3/(16 μ_3D) = 3/32
Lapk(r) = (ρ = norm(r); SMatrix{2,2,Float64,4}(1 / (4π * ρ), 0.0, 0.0, 1 / (4π * ρ)))

arc(θ1, θ2, r; c=Vec2(0.0, 0.0)) = (xx = gausslegendre(24)[1];
    legendre_coefs([c + Vec2(r * cos(θ1 + (θ2 - θ1) * (s + 1) / 2), r * sin(θ1 + (θ2 - θ1) * (s + 1) / 2)) for s in xx]))

# ---------------------------------------------------------------------------------------
# Exact reference by the divergence theorem. For a region D bounded by parameterized curves, a target
# x0 and r = y − x0, a function φ homogeneous of degree d in r has ∇·(r φ) = (d + 2) φ, so
#   ∫_D φ dA = (1/(d + 2)) ∮_∂D φ (r·n) ds,
# a 1D integral over the exact curves (QuadGK). The scalar kernels times a monomial r₁ⁱ r₂ʲ (a density
# polynomial in the physical coordinates, expanded about x0) are of this form; log ρ needs one more step,
# ∇·(r rⁱ log ρ) = (k + 2) rⁱ log ρ + rⁱ with k = i + j. This holds for unweighted panels; the weighted
# edge layer, whose density carries (1+t)^{-1/2}, is checked against the disc's closed forms instead.
# ---------------------------------------------------------------------------------------
"""Sides of a panel as τ ↦ (x, dx/dτ), τ ∈ [−1, 1], counterclockwise (J > 0)."""
panel_sides(P::Panel) = (τ -> (q = panel_map(P, τ, -1.0); (q[1], q[2])),     # bottom, s: −1 → 1
                         τ -> (q = panel_map(P, 1.0, τ); (q[1], q[3])),      # right, t: −1 → 1
                         τ -> (q = panel_map(P, -τ, 1.0); (q[1], -q[2])),    # top, s: 1 → −1
                         τ -> (q = panel_map(P, -1.0, -τ); (q[1], -q[3])))   # left, t: 1 → −1

"""Density μ(y) = Σ c y₁ᵃ y₂ᵇ, given as terms (c, a, b)."""
poly_eval(terms, y) = sum(c * y[1]^a * y[2]^b for (c, a, b) in terms)

"""
    exact_integral(P, x0, μterms; rtol=1e-13) -> SVector{4}

∫ over the (unweighted) panel P of K(x0 − y) μ(y) dA for the four scalar kernels of `Kscal`, with μ a
polynomial in the physical coordinates (terms (c, a, b)), by the divergence theorem.
"""
function exact_integral(P::Panel, x0::Vec2, μterms; rtol=1e-13)
    P.weighted && error("exact_integral: weighted panels carry (1+t)^{-1/2}; use the disc closed forms")
    rt = Dict{Tuple{Int,Int},Float64}()                         # μ in powers of r = y − x0
    for (c, a, b) in μterms, i in 0:a, j in 0:b
        rt[(i, j)] = get(rt, (i, j), 0.0) + c * binomial(a, i) * x0[1]^(a - i) * binomial(b, j) * x0[2]^(b - j)
    end
    function f(y, dy)
        r = y - x0; ρ = norm(r); rn = r[1] * dy[2] - r[2] * dy[1]          # (r·n) ds/dτ
        ρ == 0 && return zero(SVector{4,Float64})
        acc = zero(SVector{4,Float64})
        for ((i, j), c) in rt
            m = r[1]^i * r[2]^j * rn; k = i + j
            acc += c * SVector(m * (log(ρ) - 1 / (k + 2)) / (k + 2), m / (ρ * (k + 1)),
                               m * r[1] * r[2] / (ρ^2 * (k + 2)), m * r[1] * r[2] / (ρ^3 * (k + 1)))
        end
        acc
    end
    tot = zero(SVector{4,Float64})
    for side in panel_sides(P)
        τs = range(-1, 1, length=401)
        τ0 = τs[argmin([norm(side(τ)[1] - x0) for τ in τs])]          # split near the closest point
        bps = -1 < τ0 < 1 ? (-1.0, τ0, 1.0) : (-1.0, 1.0)
        tot += quadgk(τ -> f(side(τ)...), bps...; rtol=rtol, atol=1e-15)[1]
    end
    tot
end

# smooth density in the physical coordinates (degree 4)
const μphys = [(1.0, 0, 0), (0.3, 1, 0), (-0.2, 0, 1), (0.1, 1, 2), (0.05, 3, 0), (-0.07, 2, 2)]

const _star_ad = let pans = discretize_curve(θ -> (1 + 0.3cos(5θ)) * Vec2(cos(θ), sin(θ)), 10; tol=1e-12)
    build_mesh(pans, 10; edge_layers=3)[2]
end
_star_panel(f) = _star_ad.panels[findfirst(f, _star_ad.panels)]
test_panels = [
    ("straight", Panel(segment_coefs(Vec2(-0.5, -0.4), Vec2(0.6, -0.3)), segment_coefs(Vec2(-0.4, 0.5), Vec2(0.5, 0.6)), false, 0)),
    ("curved-skew", Panel(arc(-0.3, 0.5, 1.3, c=Vec2(-0.4, 0.0)), segment_coefs(Vec2(0.2, -0.05), Vec2(0.65, 0.65)), false, 0)),
    ("graded layer", _star_panel(P -> !P.weighted && P.bpanel > 0)),          # star mesh, aspect ratio ~50
    ("interior", _star_panel(P -> P.bpanel == 0)),
]

# targets: (label, kind, (s,t)) with kind :on (point x(s,t) on the panel) or :off (x(s,t) + δ·outward)
function panel_targets(P::Panel)
    T = Tuple{String,Symbol,Vec2,Float64}[]
    push!(T, ("interior", :on, Vec2(0.31, 0.27), 0.0))
    push!(T, ("near side", :on, Vec2(0.985, 0.2), 0.0))
    push!(T, ("near corner", :on, Vec2(-0.99, 0.985), 0.0))
    push!(T, ("near bottom", :on, Vec2(0.1, -0.985), 0.0))
    push!(T, ("on side", :on, Vec2(1.0, 0.3), 0.0))
    push!(T, ("on corner", :on, Vec2(-1.0, 1.0), 0.0))
    if P.weighted
        push!(T, ("on Γ", :on, Vec2(0.6, -1.0), 0.0))
        push!(T, ("on Γ near end", :on, Vec2(0.987, -1.0), 0.0))
        push!(T, ("Γ corner", :on, Vec2(-1.0, -1.0), 0.0))
    end
    for δ in (1e-1, 1e-4, 1e-8)
        push!(T, ("off side δ=$δ", :off, Vec2(1.0, -0.2), δ))
    end
    push!(T, ("off top corner δ=1e-3", :off, Vec2(1.0, 1.0), 1e-3))
    push!(T, ("off top δ=0.3", :off, Vec2(0.4, 1.0), 0.3))
    T
end

function target_point(P::Panel, kind, st, δ)
    x, xs, xt, _ = panel_map(P, st[1], st[2])
    kind == :on && return x
    # outward direction: away from the panel center
    n = st[1] == 1.0 && abs(st[2]) < 1 ? Vec2(xt[2], -xt[1]) : x - panel_center(P)
    n = n / norm(n)
    dot(n, x - panel_center(P)) < 0 && (n = -n)
    x + δ * panel_diameter(P) * n
end

"""Unit disk: K boundary arcs, K edge quads to an inner K-gon of radius rin, the K-gon split
into K triangles (centroid → vertices), each triangle into 3 quads."""
function disk_mesh(K::Int, rin::Float64)
    xx = gausslegendre(24)[1]
    bp = CurvePanel[]; ap = Panel[]
    V = [rin * Vec2(cos(2π * (k - 1) / K), sin(2π * (k - 1) / K)) for k in 1:K+1]
    for k in 1:K
        a = 2π * (k - 1) / K; b = 2π * k / K
        c = legendre_coefs([Vec2(cos(a + (b - a) * (s + 1) / 2), sin(a + (b - a) * (s + 1) / 2)) for s in xx])
        push!(bp, CurvePanel(c))
        push!(ap, Panel(c, segment_coefs(V[k], V[k+1]), true, k))
    end
    quad(a, b, c, d) = Panel(segment_coefs(a, b), segment_coefs(d, c), false, 0)
    O = Vec2(0.0, 0.0)
    for k in 1:K
        A = V[k]; B = V[k+1]; G = (O + A + B) / 3
        mOA = (O + A) / 2; mAB = (A + B) / 2; mBO = (B + O) / 2
        push!(ap, quad(O, mOA, G, mBO)); push!(ap, quad(A, mAB, G, mOA)); push!(ap, quad(B, mBO, G, mAB))
    end
    bp, ap
end

"""Smooth rule Σ_{j (≠ i)} K(xᵢ − x_j) w_j μ̃_j (dense; stands in for the FMM)."""
function smooth_apply(ad, Kf, μt::Vector{Float64}, targets; selfmode)
    out = zeros(2length(targets))
    Threads.@threads for i in eachindex(targets)
        acc = zero(Vec2)
        for j in 1:length(ad)
            selfmode && i == j && continue
            acc += Kf(targets[i] - ad.x[j]) * (ad.w[j] * Vec2(μt[2j-1], μt[2j]))
        end
        out[2i-1] = acc[1]; out[2i] = acc[2]
    end
    out
end

"""n×n grid of curved quads on [0,1]², bottom row weighted."""
function grid_mesh(n)
    f(x, y) = Vec2(x + 0.03sin(2π * y) * sin(π * x), y + 0.03sin(2π * x) * sin(π * y))
    xx = gausslegendre(8)[1]
    ap = Panel[]
    for j in 1:n, i in 1:n
        x1 = (i - 1) / n; x2 = i / n; y1 = (j - 1) / n; y2 = j / n
        B = legendre_coefs([f(x1 + (x2 - x1) * (s + 1) / 2, y1) for s in xx])
        T = legendre_coefs([f(x1 + (x2 - x1) * (s + 1) / 2, y2) for s in xx])
        push!(ap, Panel(B, T, j == 1, j == 1 ? i : 0))
    end
    ap
end
