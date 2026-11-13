# Core discretization types shared by every other file.
#
# Conventions (fixed; everything else relies on them):
#   * Ω ⊂ R² (the plane z = 0). Points are Vec2.
#   * ∂Ω = Γ is oriented counterclockwise; n is the OUTWARD unit normal, τ the unit tangent,
#     n = (τ₂, -τ₁).
#   * Every area panel is a "ruled" quad with parameters (s,t) ∈ [-1,1]²:
#         x(s,t) = ½(1-t) B(s) + ½(1+t) T(s),
#     B, T polynomial curves stored as Legendre coefficients. Orientation is such that
#     det[∂ₛx ∂ₜx] > 0.
#   * Edge panels have their bottom edge (t = -1) on Γ, with B equal to the boundary panel's
#     curve. Densities on such panels are written μ = μ̃ (1+t)^{-1/2} with μ̃ smooth; the
#     unknowns stored at nodes are μ̃, and the (1+t)^{-1/2} factor is folded into the
#     t-quadrature weights (Gauss–Jacobi α = 0, β = -1/2).
#   * Panel nodes are a tensor p×p grid, s index fastest: local index i + (j-1)p.
#   * The quadrature weight at an area node is w = wₛ wₜ J, so for a smooth f,
#     ∫_Ω f μ dA ≈ Σ f(xᵢ) μ̃ᵢ wᵢ  (μ̃ = μ on unweighted panels).

using StaticArrays, LinearAlgebra

"A point or vector of the plane, `SVector{2,Float64}`."
const Vec2 = SVector{2,Float64}
"A 2×2 matrix, `SMatrix{2,2,Float64,4}`: the form of every kernel value in the quadrature routines."
const Mat2 = SMatrix{2,2,Float64,4}

# ---------------------------------------------------------------------------------------
# 1D rules and Lagrange interpolation
# ---------------------------------------------------------------------------------------

"""1D interpolatory quadrature rule on [-1,1]; `w` includes any weight function."""
struct Rule1D
    x::Vector{Float64}
    w::Vector{Float64}
    bw::Vector{Float64}   # barycentric weights for Lagrange interpolation at x
end

function Rule1D(x::AbstractVector, w::AbstractVector)
    n = length(x)
    bw = [1 / prod(x[k] - x[j] for j in 1:n if j != k; init=1.0) for k in 1:n]
    Rule1D(collect(float.(x)), collect(float.(w)), bw)
end

"""
    gausslegendre(n) -> (x, w)

n-point Gauss–Legendre rule on [−1, 1], nodes ascending. Newton's method in θ (x = cos θ) on
f(θ) = P_n(cos θ), by the three-term recurrence, from θ = π(k − ¼)/(n + ½); the weight is
2/f'(θ)², f'(θ) = n (x P_n − P_{n−1})/sin θ. Against 256-bit references: nodes to 1.4e-16,
weights to (relative) 1.6e-14 for n ≤ 40 and 6e-13 at n = 200 (end nodes), integrals of
monomials to 1.1e-15. (Replaces FastGaussQuadrature's routine of the same name, whose weights
are within 1.2e-13 for n ≤ 200.)
"""
function gausslegendre(n::Int)
    n >= 1 || error("gausslegendre: n must be ≥ 1")
    x = zeros(n); w = zeros(n)
    function f(z, s)                               # (P_n(z), dP_n(cos θ)/dθ) at z = cos θ, s = sin θ
        p0, p1 = 1.0, z
        for j in 2:n
            p0, p1 = p1, ((2j - 1) * z * p1 - (j - 1) * p0) / j
        end
        p1, n * (z * p1 - p0) / s
    end
    for k in 1:(n + 1) ÷ 2
        θ = π * (k - 0.25) / (n + 0.5)
        for _ in 1:100
            p, dp = f(cos(θ), sin(θ)); dθ = p / dp; θ -= dθ
            abs(dθ) <= 2eps() && break
        end
        z, s = isodd(n) && k == (n + 1) ÷ 2 ? (0.0, 1.0) : (cos(θ), sin(θ))
        x[k] = -z; x[n+1-k] = z
        w[k] = w[n+1-k] = 2 / f(z, s)[2]^2
    end
    x, w
end

"p-point Gauss–Legendre rule on [−1, 1]."
gl_rule(p::Int) = Rule1D(gausslegendre(p)...)
"Gauss–Jacobi rule with weight (1+x)^{-1/2}."
gj_rule(p::Int) = Rule1D(gauss_jacobi_half(p)...)

# p-point Gauss rule for the weight (1 + x)^{-1/2} on [−1, 1], from Gauss–Legendre on 2p points: with
# x = 2s² − 1, (1 + x)^{-1/2} dx = 2√2 ds on s ∈ [0, 1], and the positive half of the 2p-point Legendre
# rule integrates the even integrand to degree 2p − 1 in x, so it is the Gauss rule. Agrees with
# FastGaussQuadrature's gaussjacobi(p, 0, −1/2) to 1e-13; that general routine cost ~6 s of compilation
# (precompile time).
function gauss_jacobi_half(p::Int)
    s, w = gausslegendre(2p)
    k = p+1:2p
    2 .* s[k] .^ 2 .- 1, 2sqrt(2) .* w[k]
end

Base.length(r::Rule1D) = length(r.x)

"""Values of the Lagrange basis polynomials of rule `r` at `x`, written into `out`."""
function lagrange_row!(out::AbstractVector, r::Rule1D, x::Real)
    n = length(r.x)
    for k in 1:n
        if x == r.x[k]
            fill!(out, 0.0); out[k] = 1.0
            return out
        end
    end
    s = 0.0
    for k in 1:n
        out[k] = r.bw[k] / (x - r.x[k])
        s += out[k]
    end
    out ./= s
    return out
end
"Values of the Lagrange basis polynomials of rule `r` at `x` (a new vector; see `lagrange_row!`)."
lagrange_row(r::Rule1D, x::Real) = lagrange_row!(zeros(length(r)), r, x)

"""Matrix mapping values at nodes of `r` to values at points `xs`."""
interp_matrix(r::Rule1D, xs::AbstractVector) =
    reduce(vcat, (lagrange_row(r, x)' for x in xs); init=zeros(0, length(r)))

"""Spectral differentiation matrix at the nodes of `r`."""
function diff_matrix(r::Rule1D)
    n = length(r); x = r.x; bw = r.bw
    D = zeros(n, n)
    for i in 1:n, j in 1:n
        i == j && continue
        D[i, j] = (bw[j] / bw[i]) / (x[i] - x[j])
    end
    for i in 1:n
        D[i, i] = -sum(D[i, j] for j in 1:n if j != i)
    end
    D
end

# ---------------------------------------------------------------------------------------
# Legendre series (used for panel geometry)
# ---------------------------------------------------------------------------------------

"""Evaluate a Legendre series Σ cₖ P_{k-1}(s) and its derivative."""
function legendre_eval(c::AbstractVector{T}, s::Real) where {T}
    n = length(c)
    p0, p1 = 1.0, s
    d0, d1 = 0.0, 1.0
    v = c[1] * p0
    dv = zero(T)
    n >= 2 && (v += c[2] * p1; dv += c[2] * d1)
    for k in 2:n-1
        p2 = ((2k - 1) * s * p1 - (k - 1) * p0) / k
        d2 = d0 + (2k - 1) * p1          # P'_{k} = P'_{k-2} + (2k-1) P_{k-1}
        v += c[k+1] * p2
        dv += c[k+1] * d2
        p0, p1 = p1, p2
        d0, d1 = d1, d2
    end
    return v, dv
end

"""Legendre coefficients of the interpolant through values `v` at Gauss–Legendre nodes."""
function legendre_coefs(v::AbstractVector{T}) where {T}
    n = length(v)
    x, w = gausslegendre(n)
    c = Vector{T}(undef, n)
    for k in 0:n-1
        acc = zero(T)
        for i in 1:n
            acc += w[i] * legendre_P(k, x[i]) * v[i]
        end
        c[k+1] = (2k + 1) / 2 * acc
    end
    c
end

"The Legendre polynomial P_k(s), by the three-term recurrence."
function legendre_P(k::Int, s::Real)
    k == 0 && return 1.0
    p0, p1 = 1.0, float(s)
    for j in 2:k
        p0, p1 = p1, ((2j - 1) * s * p1 - (j - 1) * p0) / j
    end
    p1
end

"""Legendre coefficients of the derivative series: b_k = (2k+1) Σ_{j>k, j-k odd} c_j."""
function legendre_deriv_coefs(c::AbstractVector{T}) where {T}
    n = length(c)
    n == 1 && return [zero(T)]
    b = [zero(T) for _ in 1:n-1]
    for k in 0:n-2, j in k+1:2:n-1
        b[k+1] += (2k + 1) * c[j+1]
    end
    b
end

"""Legendre coefficients of the segment from a (s=-1) to b (s=+1)."""
segment_coefs(a::Vec2, b::Vec2) = [(a + b) / 2, (b - a) / 2]

# ---------------------------------------------------------------------------------------
# Panels
# ---------------------------------------------------------------------------------------

"""Boundary panel: a piece of Γ, γ(s) = Σ cₖ P_{k-1}(s), s ∈ [-1,1], traversed with the film on the left
(outer curves counterclockwise, hole curves clockwise)."""
struct CurvePanel
    c::Vector{Vec2}
end
"(γ(s), γ′(s)) of the boundary panel `P` at its parameter s ∈ [−1, 1]."
curve_eval(P::CurvePanel, s::Real) = legendre_eval(P.c, s)   # (γ, γ')

"""Ruled area panel x(s,t) = ½(1-t)B(s) + ½(1+t)T(s)."""
struct Panel
    B::Vector{Vec2}
    T::Vector{Vec2}
    weighted::Bool   # bottom edge on Γ, density carries (1+t)^{-1/2}
    bpanel::Int      # edge column: index of the boundary panel this edge layer sits over (0: interior)
end

"""Return (x, ∂ₛx, ∂ₜx, J) at parameters (s,t)."""
@inline function panel_map(P::Panel, s::Real, t::Real)
    b, db = legendre_eval(P.B, s)
    u, du = legendre_eval(P.T, s)
    x = 0.5 * (1 - t) * b + 0.5 * (1 + t) * u
    xs = 0.5 * (1 - t) * db + 0.5 * (1 + t) * du
    xt = 0.5 * (u - b)
    J = xs[1] * xt[2] - xs[2] * xt[1]
    return x, xs, xt, J
end

"""Characteristic size of a panel (max distance between corners)."""
function panel_diameter(P::Panel)
    c = (panel_map(P, -1, -1)[1], panel_map(P, 1, -1)[1], panel_map(P, 1, 1)[1], panel_map(P, -1, 1)[1])
    maximum(norm(c[i] - c[j]) for i in 1:4 for j in i+1:4)
end
"The point x(0, 0) of the area panel `P`."
panel_center(P::Panel) = panel_map(P, 0.0, 0.0)[1]

# ---------------------------------------------------------------------------------------
# Discretizations
# ---------------------------------------------------------------------------------------

"""
    BoundaryDisc(panels, p; pstart, cstart)

The boundary Γ on its panels, with p Gauss–Legendre nodes per panel: nodes `x`, arclength weights `w`, unit tangents
`τ`, outward normals `n` and signed curvatures `κ` (> 0 where Ω is locally convex). Panel k owns the nodes
`panel_nodes(bd, k)`. Several drops and curves: drop i owns the panels `drop_panels(bd, i)`, closed curve c the panels
`curve_panels(bd, c)` (each drop: its outer curve first, then its holes). Built by `build_mesh`.
"""
struct BoundaryDisc
    p::Int
    rule::Rule1D                  # Gauss–Legendre, p nodes
    panels::Vector{CurvePanel}
    x::Vector{Vec2}               # nodes
    w::Vector{Float64}            # arclength weights (wᵢ |γ'|)
    τ::Vector{Vec2}               # unit tangent
    n::Vector{Vec2}               # outward unit normal
    κ::Vector{Float64}            # signed curvature (> 0 where the film is locally convex)
    pstart::Vector{Int}           # drop i owns panels pstart[i]:pstart[i+1]-1
    cstart::Vector{Int}           # closed curve c is panels cstart[c]:cstart[c+1]-1 (refines pstart;
                                  # each drop: outer curve first, then its holes)
end

function BoundaryDisc(panels::Vector{CurvePanel}, p::Int; pstart::Vector{Int}=[1, length(panels) + 1],
                      cstart::Vector{Int}=pstart)
    r = gl_rule(p)
    x = Vec2[]; w = Float64[]; τ = Vec2[]; n = Vec2[]; κ = Float64[]
    for P in panels
        dc = legendre_deriv_coefs(P.c)
        for (i, s) in enumerate(r.x)
            g, dg = curve_eval(P, s)
            _, dd = legendre_eval(dc, s)
            sp = norm(dg)
            t = dg / sp
            push!(x, g); push!(w, r.w[i] * sp); push!(τ, t); push!(n, Vec2(t[2], -t[1]))
            push!(κ, (dg[1] * dd[2] - dg[2] * dd[1]) / sp^3)
        end
    end
    _check_pstart(pstart, length(panels)); _check_pstart(cstart, length(panels))
    issubset(pstart, cstart) || error("cstart = $cstart must refine pstart = $pstart (drops are unions of whole curves)")
    BoundaryDisc(p, r, panels, x, w, τ, n, κ, pstart, cstart)
end

"The number of panels of a `BoundaryDisc` or an `AreaDisc`."
npanels(d::BoundaryDisc) = length(d.panels)
Base.length(d::BoundaryDisc) = length(d.x)
"The indices of the nodes of panel `k` (p of them on Γ, p² on an area panel)."
panel_nodes(d::BoundaryDisc, k::Int) = (k-1)*d.p+1:k*d.p

"""
    AreaDisc(panels, p; pstart)

The area mesh of Ω: ruled `Panel`s with a p×p tensor grid of nodes each (s index fastest; panel k owns
`panel_nodes(ad, k)`), the nodes `x`, their parameters `st`, Jacobians `J` and quadrature weights `w` = wₛ wₜ J. On a
weighted panel (an edge panel, its bottom edge on Γ) a density is stored as μ̃ = μ (1 + t)^{1/2}, the t-rule is
Gauss–Jacobi and its weight is folded into `w`; `wfac` = (1 + t)^{−1/2} there (1 elsewhere) gives μ = μ̃ · wfac.
Drop i owns the panels `drop_panels(ad, i)`. Built by `build_mesh`.
"""
struct AreaDisc
    p::Int
    rs::Rule1D                    # s-rule (Gauss–Legendre)
    rt::Rule1D                    # t-rule, unweighted panels (Gauss–Legendre)
    rtw::Rule1D                   # t-rule, weighted panels (Gauss–Jacobi, (1+t)^{-1/2})
    panels::Vector{Panel}
    x::Vector{Vec2}               # nodes
    w::Vector{Float64}            # weights wₛ wₜ J (Jacobi weight included)
    st::Vector{Vec2}              # (s,t) parameters of each node
    J::Vector{Float64}            # Jacobian at each node
    wfac::Vector{Float64}         # μ = μ̃ * wfac  ((1+t)^{-1/2} on weighted panels, else 1)
    pstart::Vector{Int}           # drop i owns panels pstart[i]:pstart[i+1]-1
end

function AreaDisc(panels::Vector{Panel}, p::Int; pstart::Vector{Int}=[1, length(panels) + 1])
    rs = gl_rule(p); rt = gl_rule(p); rtw = gj_rule(p)
    N = length(panels) * p^2
    x = Vector{Vec2}(undef, N); w = zeros(N); st = Vector{Vec2}(undef, N)
    J = zeros(N); wfac = ones(N)
    for (k, P) in enumerate(panels)
        tr = P.weighted ? rtw : rt
        for j in 1:p, i in 1:p
            idx = (k - 1) * p^2 + i + (j - 1) * p
            s, t = rs.x[i], tr.x[j]
            xx, _, _, JJ = panel_map(P, s, t)
            JJ > 0 || error("panel $k has non-positive Jacobian at (s,t)=($s,$t)")
            x[idx] = xx; st[idx] = Vec2(s, t); J[idx] = JJ
            w[idx] = rs.w[i] * tr.w[j] * JJ
            P.weighted && (wfac[idx] = 1 / sqrt(1 + t))
        end
    end
    _check_pstart(pstart, length(panels))
    AreaDisc(p, rs, rt, rtw, panels, x, w, st, J, wfac, pstart)
end

npanels(d::AreaDisc) = length(d.panels)
Base.length(d::AreaDisc) = length(d.x)
panel_nodes(d::AreaDisc, k::Int) = (k-1)*d.p^2+1:k*d.p^2
"The t-rule of area panel `k`: Gauss–Jacobi on a weighted panel, Gauss–Legendre otherwise."
trule(d::AreaDisc, k::Int) = d.panels[k].weighted ? d.rtw : d.rt

# ---------------------------------------------------------------------------------------
# Several drops: each drop's panels are contiguous; `pstart` holds the offsets
# ---------------------------------------------------------------------------------------

function _check_pstart(pstart, n)
    (length(pstart) >= 2 && pstart[1] == 1 && pstart[end] == n + 1 && all(diff(pstart) .> 0)) ||
        error("pstart = $pstart is not a partition of 1:$n into nonempty ranges")
end

const _Disc = Union{BoundaryDisc,AreaDisc}
_nper(d::BoundaryDisc) = d.p
_nper(d::AreaDisc) = d.p^2
"The number of drops (disjoint components of Ω) of a discretization."
ndrops(d::_Disc) = length(d.pstart) - 1
"The panels of drop `i`."
drop_panels(d::_Disc, i::Int) = d.pstart[i]:d.pstart[i+1]-1
"The nodes of drop `i`."
drop_nodes(d::_Disc, i::Int) = (d.pstart[i]-1)*_nper(d)+1:(d.pstart[i+1]-1)*_nper(d)
"""Drop index of each panel."""
panel_drop(d::_Disc) = reduce(vcat, [fill(i, length(drop_panels(d, i))) for i in 1:ndrops(d)])
"""Drop index of each node."""
node_drop(d::_Disc) = reduce(vcat, [fill(i, length(drop_nodes(d, i))) for i in 1:ndrops(d)])

# closed curves of a boundary discretization (a drop with holes has several)
"The number of closed curves of Γ (outer curves and holes)."
ncurves(d::BoundaryDisc) = length(d.cstart) - 1
"The panels of closed curve `c`."
curve_panels(d::BoundaryDisc, c::Int) = d.cstart[c]:d.cstart[c+1]-1
"The nodes of closed curve `c`."
curve_nodes(d::BoundaryDisc, c::Int) = (d.cstart[c]-1)*d.p+1:(d.cstart[c+1]-1)*d.p
"""Curves of drop i (outer first)."""
drop_curves(d::BoundaryDisc, i::Int) = findfirst(==(d.pstart[i]), d.cstart):findfirst(==(d.pstart[i+1]), d.cstart)-1
"""Next and previous panel of each panel along its own closed curve, for curve offsets `cstart`."""
function cyclic_maps(cstart::AbstractVector{Int})
    N = cstart[end] - 1
    nxt = zeros(Int, N); prv = zeros(Int, N)
    for c in 1:length(cstart)-1
        a, b = cstart[c], cstart[c+1] - 1
        for k in a:b
            nxt[k] = k == b ? a : k + 1
            prv[k] = k == a ? b : k - 1
        end
    end
    nxt, prv
end
cyclic_maps(d::BoundaryDisc) = cyclic_maps(d.cstart)

"""
Matrix (npts × p²) evaluating the tensor interpolant of panel `k`'s nodal values at
parameter points `pts` (vector of (s,t)). For weighted panels this interpolates μ̃.
"""
function panel_interp_matrix(d::AreaDisc, k::Int, pts::AbstractVector)
    p = d.p; rt = trule(d, k)
    M = zeros(length(pts), p^2)
    ls = zeros(p); lt = zeros(p)
    for (m, q) in enumerate(pts)
        lagrange_row!(ls, d.rs, q[1]); lagrange_row!(lt, rt, q[2])
        for j in 1:p, i in 1:p
            M[m, i + (j-1)*p] = ls[i] * lt[j]
        end
    end
    M
end
