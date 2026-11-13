# Shared by test_quadrature.jl (regression) and validation/val_line_quad.jl (convergence
# report): API names, test curves, densities and QuadGK references.

using OpenSurfaces, LinearAlgebra, StaticArrays, SparseArrays, Test, QuadGK
using OpenSurfaces: gausslegendre, _lq_G2, _lq_T2, _lq_flat, _lq_vecs, _lq_log_moments
const LineQuad_ = LineQuad
G2t(r, η=1.0) = _lq_G2(r, η)
T2t(r, n) = _lq_T2(r, n)
flat(v) = _lq_flat(v); vecs(v) = _lq_vecs(v)

# ---------------------------------------------------------------------------------------
# Geometry and references
# ---------------------------------------------------------------------------------------

function curve_disc(f, K, p; nfit=24)
    xf, _ = gausslegendre(nfit)
    pans = CurvePanel[]
    for k in 1:K
        a = 2π * (k - 1) / K; b = 2π * k / K
        v = [Vec2(f(a + (b - a) * (s + 1) / 2)...) for s in xf]
        push!(pans, CurvePanel(legendre_coefs(v)))
    end
    BoundaryDisc(pans, p)
end
circle(a=1.0) = θ -> (a * cos(θ), a * sin(θ))
ellipse(a=1.5, b=0.7) = θ -> (a * cos(θ), b * sin(θ))
star(ε=0.3, m=5) = θ -> ((1 + ε * cos(m * θ)) * cos(θ), (1 + ε * cos(m * θ)) * sin(θ))
smooth_density(bd) = [Vec2(sin(3x[1]) + x[2]^2, cos(2x[2]) - x[1]) for x in bd.x]
normal_at(P, s) = (dg = curve_eval(P, s)[2]; Vec2(dg[2], -dg[1]) / norm(dg))

function closest_s(P, x)
    ss = range(-1, 1, length=401)
    s = ss[argmin([norm(x - curve_eval(P, t)[1]) for t in ss])]
    dc = legendre_deriv_coefs(P.c)
    for _ in 1:30
        g, dg = curve_eval(P, s); ddg = legendre_eval(dc, s)[2]
        s = clamp(s - dot(g - x, dg) / (dot(dg, dg) + dot(g - x, ddg)), -1, 1)
    end
    s
end

# ∫_panel K(x−γ(s)) ν_interp(s) |γ'(s)| ds by adaptive Gauss–Kronrod (breakpoint at the
# closest point, or at `sb` when given, e.g. the target's own parameter).
function ref_panel(K, bd, k, x, ν; sb=nothing)
    P = bd.panels[k]; idx = panel_nodes(bd, k); ℓ = zeros(bd.p)
    f = s -> begin
        g, dg = curve_eval(P, s); lagrange_row!(ℓ, bd.rule, s)
        nu = sum(ℓ[j] * ν[idx[j]] for j in 1:bd.p)
        r = x - g
        iszero(r) ? zero(Vec2) : K(r) * nu * norm(dg)
    end
    quadgk(f, -1.0, sb === nothing ? closest_s(P, x) : sb, 1.0; rtol=1e-14, atol=1e-16, maxevals=10^7)[1]
end
ref_total(K, bd, x, ν) = sum(ref_panel(K, bd, k, x, ν) for k in 1:npanels(bd))
function ref_total_node(K, bd, it, ν)
    p = bd.p; k0 = (it - 1) ÷ p + 1; si = bd.rule.x[it-(k0-1)*p]
    sum(ref_panel(K, bd, k, bd.x[it], ν; sb=(k == k0 ? si : nothing)) for k in 1:npanels(bd))
end

# BigFloat curve evaluation (the PV traction integrand cancels catastrophically near the target)
function big_curve(P, s::BigFloat)
    c = P.c; n = length(c)
    cb(k) = [BigFloat(c[k][1]), BigFloat(c[k][2])]
    p0, p1 = one(BigFloat), s
    v = cb(1); n >= 2 && (v .+= cb(2) .* p1)
    for k in 2:n-1
        p2 = ((2k - 1) * s * p1 - (k - 1) * p0) / k
        v .+= cb(k + 1) .* p2
        p0, p1 = p1, p2
    end
    v
end
function ref_traction_pv(bd, it, ν)
    p = bd.p; k0 = (it - 1) ÷ p + 1; si = bd.rule.x[it-(k0-1)*p]
    n = bd.n[it]; tot = zero(Vec2)
    for k in 1:npanels(bd)
        P = bd.panels[k]; idx = panel_nodes(bd, k); ℓ = zeros(p)
        f = s -> begin
            g, dg = curve_eval(P, s); lagrange_row!(ℓ, bd.rule, s)
            nu = sum(ℓ[j] * ν[idx[j]] for j in 1:p)
            if k == k0 && abs(s - si) < 1e-3
                setprecision(BigFloat, 200) do
                    r = big_curve(P, BigFloat(si)) .- big_curve(P, BigFloat(s))
                    ρ2 = r[1]^2 + r[2]^2
                    iszero(ρ2) && return -(bd.κ[it] / (2π)) * (bd.τ[it] * bd.τ[it]') * nu * norm(dg)
                    sc = -(r[1] * n[1] + r[2] * n[2]) / (BigFloat(π) * ρ2^2)
                    Mat2(Float64(sc * r[1]^2), Float64(sc * r[1] * r[2]), Float64(sc * r[1] * r[2]),
                         Float64(sc * r[2]^2)) * nu * norm(dg)
                end
            else
                T2t(bd.x[it] - g, n) * nu * norm(dg)
            end
        end
        tot += quadgk(f, -1.0, 1.0; rtol=1e-13, atol=1e-15)[1]
    end
    tot
end

# pressure kernel as the first row of a Mat2: (1/2π)(r·f)/ρ²
P2row(r) = (c = 1 / (2π * dot(r, r)); Mat2(c * r[1], 0.0, c * r[2], 0.0))

# interior targets near node `it` (distance d·L along −n) and at a few panel positions
function near_targets(bd, lq, ks, ss, ds; side=-1)
    tg = Vec2[]
    for k in ks, s in ss, d in ds
        g = curve_eval(bd.panels[k], s)[1]
        push!(tg, g + side * d * lq.len[k] * normal_at(bd.panels[k], s))
    end
    tg
end
