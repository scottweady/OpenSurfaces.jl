# Local (singular / nearly singular) quadrature for area potentials over ruled quad panels.
#
#   ∫_panel K(x − y) μ(y) dA_y ≈ Σ_k W[k] μ̃_k ,   μ = μ̃ (1+t)^{-1/2} on weighted panels.
#
# W[k] (2×2) = Σ_q K(x − y_q) ℓ_k(s_q,t_q) ω_q J(s_q,t_q) over a target-adapted rule in
# parameter space; ℓ_k is the tensor Lagrange basis of the panel's nodes (μ̃ interpolant).
# The kernel is a black box K(i, r)::SMatrix{2,2} (i = target index, r = x − y) or K(r).
#
# Rules (all built in parameter space, kernel-agnostic, adapted to the panel geometry):
#  * target ON the panel (self, side, corner): polar/Duffy rule on the whole square with apex
#    at the target. Each of the (up to) 4 triangles apex→side is mapped (u,v) ∈ [0,1]²,
#    y = A + u (P1 + v (P2 − P1) − A), Jacobian u·|det|. The Jacobian u cancels 1/ρ.
#      - u: composite rule [0,u1] (u = u1 w⁴, handles u log u for log kernels) ∪ [u1,1] (GL).
#      - v: composite GL, refined adaptively (Bernstein-ellipse criterion) toward the complex
#        singularities of the kernel along the base, found by linearisation at the apex (u→0)
#        and by complex Newton on the u = 1 slice. Handles apexes arbitrarily close to sides.
#    Weighted panels, apex with t0 > −1: coordinates (s, τ), t = −1 + 2τ², which removes the
#    (1+t)^{-1/2} weight exactly (dt (1+t)^{-1/2} = 2√2 dτ).
#    Weighted panels, apex on the bottom edge t0 = −1 (targets on Γ): coordinates (s,t), three
#    triangles; the weight factorises as (1+t)^{-1/2} = (2uv)^{-1/2} (side triangles) or
#    (2u)^{-1/2} (top triangle); u^{-1/2} is absorbed in the radial rule (u = u1 w⁴) and
#    v^{-1/2} by v = z².
#  * target OFF the panel: adaptive quadtree in (s,t); a box is accepted with a q×q tensor
#    rule (q from a Bernstein estimate of the physical distance, ≤ qmax_box), boxes touching
#    t = −1 on weighted panels use Gauss–Jacobi in t.
#
# Assembly (`area_correction`): sparse matrix of (local weights − smooth-rule weights) over
# all near (target, panel) pairs, interleaved 2×2 layout.

using LinearAlgebra, SparseArrays, StaticArrays

# ---------------------------------------------------------------------------------------
# Options / rules
# ---------------------------------------------------------------------------------------

struct _QRule
    x::Vector{Float64}
    w::Vector{Float64}
end

"""
    AreaQuadOpts(p; tol=DEFAULT_TOL, near_factor=nothing, ...)

Parameters of the local area quadrature. `tol` is the target accuracy (relative to the size
of a panel's contribution). `near_factor`: a (target, panel) pair is corrected when
|x − c_k| < near_factor · R_k (c_k panel center, R_k its circumradius); by default it is
derived from `p` and `tol` such that the p×p smooth rule is accurate to ≈ tol outside, and at least 1.2 (targets
near a neighboring panel's corner).

The polar rule's sizes also have floors set by the Lagrange basis: its
degree along a line is p − 1 in s and t on plain panels, but 2(p − 1) in σ or z on weighted panels
(t = −1 + 2σ², v = z²), up to 3(p − 1) along a radial line. The floors integrate the basis to ≈ tol/10: each is
its p = 10 size for ≈ 1e-10 plus the measured extra points per digit (2.2 digits per
side point, 0.9 per inner radial point). The adaptive side rule adds ⌈d/2⌉ points to the kernel's estimate
`_qneed(ρ, qtol)`, d the basis degree along the side: Gauss–Legendre is exact to degree 2q − 1, so the basis factor
spends d/2 of the ρ^{−2q} rate (without it the weights miss by up to 24 × tol at tol 1e-6 and 240 × tol at 1e-8).
With the shift the side rule meets ≈ tol/10 at its own `qtol_v` = tol/2
(10 × `qtol`, which the off-panel boxes keep). `floors = false` restores the tol-only sizes, without either.
"""
struct AreaQuadOpts
    p::Int
    tol::Float64
    qtol::Float64           # tolerance used in the ρ^{-2q} estimates
    qmin_v::Int             # side rule floor: plain panels, horizontal sides of weighted panels
    qmin_v2::Int            # side rule floor: vertical sides of weighted panels (degree 2(p − 1) in σ or z)
    qmax_v::Int
    qmin_box::Int
    qmax_box::Int
    maxdepth::Int
    maxpanels_v::Int
    gl::Vector{_QRule}      # Gauss–Legendre on [-1,1], indexed by q
    gj::Vector{_QRule}      # Gauss–Jacobi (0,-1/2) on [-1,1], indexed by q
    ulog::_QRule            # radial rule on [0,1] for analytic + u log u integrands (plain panels)
    utau::_QRule            # same, weighted panels with the apex inside (degree 3(p − 1) along a ray)
    uhalf::_QRule           # same, weight u^{-1/2} included (weighted panels, apex on Γ)
    near_factor::Float64
    slice_us::Vector{Float64}  # u values of the slices used for singularity detection
    basis_shift::Bool       # side rule: add ⌈d/2⌉ for the basis degree d along the side
    qtol_v::Float64         # side rule's ρ^{-2q} estimate (on-panel targets)
end

function AreaQuadOpts(p::Int; tol::Real=DEFAULT_TOL, near_factor=nothing, qmax_v::Int=16,
                      qmax_box::Int=18, maxdepth::Int=48, qradial=nothing,
                      slice_us=(0.5, 1.0), log_kernel::Bool=true,
                      radial_power=nothing, floors::Bool=true)
    tol = Float64(tol)
    qtol = tol / 20
    nd = log10(1 / tol)
    # radial rule sizes from tol (for the kernel), raised to the degree floors (for the basis); `qradial` overrides both
    q1, q2 = log_kernel ? (clamp(ceil(Int, 1.4nd), 6, 24), clamp(ceil(Int, 1.2nd), 8, 20)) :
                          (clamp(ceil(Int, 0.6nd), 4, 12), clamp(ceil(Int, 1.1nd), 6, 20))
    # degree floors for a basis error ≈ tol/10, i.e. D digits: (size at the reference digits Dref) + points per digit
    D = nd + 1
    fl(base, Dref, slope) = floors ? base + ceil(Int, (D - Dref) / slope) : 0
    o2 = !floors ? -typemax(Int) ÷ 2 : D <= 7.5 ? -2 : 0      # the outer radial part: 2 points fewer up to ~1e-7
    qr = qradial === nothing ? (max(q1, fl(2p - 2, 10.2, 0.92)), max(q2, p + 4 + o2)) : qradial
    qt = qradial === nothing ? (max(q1, fl(2p, 10.6, 0.92)), max(q2, p + 6 + o2)) : qradial
    kpow = radial_power === nothing ? (log_kernel ? (4, 4) : (1, 2)) : radial_power
    ulog = _radial_rule(qr..., 0.25, kpow[1], 0.0)
    utau = _radial_rule(qt..., 0.25, kpow[1], 0.0)
    uhalf = _radial_rule(qr..., 0.25, kpow[2], -0.5)
    qmx = max(qmax_v, qmax_box, p + 2)
    gl = [_QRule(gausslegendre(q)...) for q in 1:qmx]
    gj = [_QRule(gauss_jacobi_half(q)...) for q in 1:qmx]
    if near_factor === nothing
        ρ = (1 / tol)^(1 / (2p))
        d = (ρ - 1 / ρ) / 2              # needed distance (half-lengths, parallel direction)
        near_factor = _near_floor((1 + d) / sqrt(2))
    end
    qmin_v = max(4, (p + 3) ÷ 2, fl(p - 1, 10.5, 2.2))
    qmin_v2 = clamp(fl(p + 2, 10.6, 2.4), qmin_v, qmax_v)
    qmin_box = max(4, (p + 3) ÷ 2)
    AreaQuadOpts(p, tol, qtol, qmin_v, qmin_v2, qmax_v, qmin_box, qmax_box, maxdepth, 400,
                 gl, gj, ulog, utau, uhalf, Float64(near_factor), collect(Float64, slice_us), floors,
                 floors ? 10qtol : qtol)
end

"""
Composite radial rule for ∫_0^1 g(u) u^α du: [0,u1] with u = u1 w^k (GL in w), [u1,1] GL.
k = 4 makes u^α (1 + u log u) smooth enough for log kernels; k = 1 (α = 0) / 2 (α = −1/2)
keeps polynomial exactness for kernels without a logarithm.
"""
function _radial_rule(q1::Int, q2::Int, u1::Float64, k::Int, α::Float64)
    x, w = gausslegendre(q1); wv = (x .+ 1) ./ 2; ww = w ./ 2
    ua = u1 .* wv .^ k
    wa = u1^(1 + α) * k .* wv .^ (k * (1 + α) - 1) .* ww
    x2, w2 = gausslegendre(q2)
    ub = u1 .+ (1 - u1) .* (x2 .+ 1) ./ 2
    wb = (1 - u1) .* w2 ./ 2 .* ub .^ α
    _QRule(vcat(ua, ub), vcat(wa, wb))
end

# The near zone's floor: at loose tol or high p the estimate falls below 1, and targets at a neighbor's shared corner
# (within its circumradius of its center, but at distance ≈ 0 from it) got the smooth rule (errors up to 81 × tol).
# 1.2 keeps p = 10, tol 1e-6 (1.235) as it was.
_near_floor(f) = max(f, 1.2)

@inline _qneed(ρ::Float64, qtol::Float64) = ρ <= 1 ? typemax(Int) : ceil(Int, log(1 / qtol) / (2log(ρ)))

"Bernstein-ellipse parameter of the complex point z relative to the interval [a,b]."
@inline function _bernstein(z::ComplexF64, a::Float64, b::Float64)
    x = (2z - (a + b)) / (b - a)
    r = abs(x + sqrt(x - 1) * sqrt(x + 1))
    return r < 1 ? 1 / r : r
end

# ---------------------------------------------------------------------------------------
# Workspace
# ---------------------------------------------------------------------------------------

# V: the accumulated kernel values, SVector{4,Float64} for the 2×2 form (column-major entries) or SVector{M,Float64}
# for M scalar components (`area_correction(...; ncomp = M)`)
mutable struct _AQWork{V}
    p::Int
    W::Matrix{V}                 # p × p (i_s, j_t): the kernel's entries
    acc::Vector{V}               # p
    ls::Vector{Float64}
    lt::Vector{Float64}
    vx::Vector{Float64}          # current 1D (v) rule
    vw::Vector{Float64}
    lx::Vector{Float64}          # current line: varying coordinate, weights
    lw::Vector{Float64}
    ld::Vector{Float64}          # current line: varying coordinate minus apex coordinate
    stk::Vector{NTuple{2,Float64}}
    sings::Vector{ComplexF64}
    bstk::Vector{NTuple{5,Float64}}
    LS::Matrix{Float64}          # box s-Lagrange values, stored q × p (column i = ℓ_i at the q s-nodes)
    kb::Vector{V}                # qmax: weighted kernel values along one box row
    tb::Vector{V}                # p × qmax: s-contracted rows of a box (i fastest)
    LT::Vector{Float64}          # box t-Lagrange values, stored q × p
    lk::Vector{V}                # polar lines: weighted kernel values (grown as needed)
    lL::Vector{Float64}          # polar lines: Lagrange values, stored n × p (grown as needed)
    bB::Vector{NTuple{4,Vec2}}   # box s-node geometry (B, B', T, T')
    bsw::Vector{Float64}
    npts::Int                    # statistics: number of kernel evaluations
end

_AQWork(p::Int, qmax::Int) = _AQWork{SVector{4,Float64}}(p, qmax)
function _AQWork{V}(p::Int, qmax::Int) where {V}
    _AQWork{V}(p, zeros(V, p, p), zeros(V, p), zeros(p), zeros(p), Float64[], Float64[],
               Float64[], Float64[], Float64[], NTuple{2,Float64}[], ComplexF64[], NTuple{5,Float64}[],
               zeros(p, qmax), zeros(V, qmax), zeros(V, p * qmax), zeros(qmax * p),
               V[], Float64[], Vector{NTuple{4,Vec2}}(undef, qmax), zeros(qmax), 0)
end

# A kernel value times the weight c, as the accumulated type V: the 2×2 form's column-major entries (the kernel
# returns anything an SMatrix{2,2} takes), or M components (the kernel wrapped by `_Components`, an SVector{M}).
@inline function _kv(::Type{SVector{4,Float64}}, K, c::Float64)
    Kw = SMatrix{2,2,Float64,4}(K) * c
    SVector{4,Float64}(Kw[1], Kw[2], Kw[3], Kw[4])
end
@inline _kv(::Type{SVector{M,Float64}}, K::SVector{M,Float64}, c::Float64) where {M} = K * c
@inline _kv(::Type{SVector{4,Float64}}, K::SVector{4,Float64}, c::Float64) = K * c

"""A kernel of M scalar components (scalar, tuple or vector of length M) as K(i, r)::SVector{M,Float64}."""
struct _Components{M,F}
    k::F
end
@inline (C::_Components{M})(i::Int, r::Vec2) where {M} = SVector{M,Float64}(C.k(i, r))

"""
Lagrange basis values of rule `r` at x (first barycentric form, no divisions):
ℓ_k(x) = bw_k Π_{j≠k} (x − x_j), via prefix/suffix products.
"""
@inline function _lag!(out::Vector{Float64}, r::Rule1D, x::Float64)
    n = length(r.x)
    xs = r.x; bw = r.bw
    pre = 1.0
    @inbounds for k in 1:n
        out[k] = pre
        pre *= x - xs[k]
    end
    suf = 1.0
    @inbounds for k in n:-1:1
        out[k] *= suf * bw[k]
        suf *= x - xs[k]
    end
    out
end

"Copy of a panel with negligible trailing Legendre coefficients removed (cheaper evaluation)."
function _trim_panel(P::Panel)
    function trim(c)
        m = maximum(norm, c)
        n = length(c)
        while n > 2 && norm(c[n]) <= 1e-17 * m
            n -= 1
        end
        c[1:n]
    end
    Panel(trim(P.B), trim(P.T), P.weighted, P.bpanel)
end

# ---------------------------------------------------------------------------------------
# Line accumulation:  W[i,j] += Σ_points vec(K) ω J ℓ_i(s) ℓ_j(t)
# ---------------------------------------------------------------------------------------

"""Line buffers for n points (grown on demand)."""
@inline function _line_bufs!(wk, n::Int)
    length(wk.lk) < n && resize!(wk.lk, n)
    length(wk.lL) < n * wk.p && resize!(wk.lL, n * wk.p)
    wk.lk, wk.lL
end

"""
acc[i] = Σ_{m ≤ n} kb[m] L[m,i] (i = 1..p; L stored n × p, column-major in a vector).
Accumulators stay in registers, two coefficients and two partial sums at a time (independent
FMA chains); much faster than a load/store of acc per quadrature point.
"""
@inline function _contract!(acc::Vector{V}, kb::Vector{V}, L::Vector{Float64}, n::Int, p::Int) where {V}
    i = 1
    @inbounds while i < p
        oi = (i - 1) * n; oj = oi + n
        c1 = zero(V); c2 = c1; d1 = c1; d2 = c1
        m = 1
        while m < n
            k1 = kb[m]; k2 = kb[m+1]
            c1 = muladd(k1, L[oi+m], c1); d1 = muladd(k2, L[oi+m+1], d1)
            c2 = muladd(k1, L[oj+m], c2); d2 = muladd(k2, L[oj+m+1], d2)
            m += 2
        end
        if m == n
            c1 = muladd(kb[m], L[oi+m], c1); c2 = muladd(kb[m], L[oj+m], c2)
        end
        acc[i] = c1 + d1; acc[i+1] = c2 + d2
        i += 2
    end
    @inbounds if i == p
        c1 = zero(V); oi = (i - 1) * n
        for m in 1:n
            c1 = muladd(kb[m], L[oi+m], c1)
        end
        acc[i] = c1
    end
    acc
end

# Legendre recurrence constants (2k+1)/(k+1), k/(k+1), 2k+1: tabulated, since the integer
# divisions were a large part of _geom_dd (bitwise the same values).
const _NLEG = 64
const _LEG_A = [(2k + 1) / (k + 1) for k in 1:_NLEG]
const _LEG_C = [k / (k + 1) for k in 1:_NLEG]
const _LEG_2K1 = [Float64(2k + 1) for k in 1:_NLEG]

"""
Combined evaluation at s = s0 + Δ of B, B', (B(s)−B(s0))/Δ and the same for T (one pass).
"""
@inline function _geom_dd(P::Panel, s0::Float64, Δ::Float64)
    cB = P.B; cT = P.T
    nB = length(cB); nT = length(cT); n = max(nB, nT)
    s = s0 + Δ
    ps0 = 1.0; ps1 = s          # P_k(s)
    dp0 = 0.0; dp1 = 1.0        # P_k'(s)
    q0 = 1.0; q1 = s0           # P_k(s0)
    d0 = 0.0; d1 = 1.0          # divided differences
    b = cB[1]; db = zero(Vec2); DB = zero(Vec2)
    u = cT[1]; du = zero(Vec2); DT = zero(Vec2)
    if nB >= 2
        b += cB[2] * ps1; db += cB[2]; DB += cB[2]
    end
    if nT >= 2
        u += cT[2] * ps1; du += cT[2]; DT += cT[2]
    end
    n - 2 > _NLEG && return _geom_dd_long(P, s0, Δ)
    @inbounds for k in 1:n-2
        a = _LEG_A[k]; c = _LEG_C[k]
        ps2 = a * s * ps1 - c * ps0
        dp2 = dp0 + _LEG_2K1[k] * ps1
        q2 = a * s0 * q1 - c * q0
        d2 = a * (s * d1 + q1) - c * d0
        if k + 2 <= nB
            cc = cB[k+2]; b += cc * ps2; db += cc * dp2; DB += cc * d2
        end
        if k + 2 <= nT
            cc = cT[k+2]; u += cc * ps2; du += cc * dp2; DT += cc * d2
        end
        ps0, ps1 = ps1, ps2; dp0, dp1 = dp1, dp2; q0, q1 = q1, q2; d0, d1 = d1, d2
    end
    return b, db, DB, u, du, DT
end

"""`_geom_dd` for panels with more than _NLEG + 2 coefficients (untabulated constants)."""
@noinline function _geom_dd_long(P::Panel, s0::Float64, Δ::Float64)
    cB = P.B; cT = P.T
    nB = length(cB); nT = length(cT); n = max(nB, nT)
    s = s0 + Δ
    ps0 = 1.0; ps1 = s          # P_k(s)
    dp0 = 0.0; dp1 = 1.0        # P_k'(s)
    q0 = 1.0; q1 = s0           # P_k(s0)
    d0 = 0.0; d1 = 1.0          # divided differences
    b = cB[1]; db = zero(Vec2); DB = zero(Vec2)
    u = cT[1]; du = zero(Vec2); DT = zero(Vec2)
    if nB >= 2
        b += cB[2] * ps1; db += cB[2]; DB += cB[2]
    end
    if nT >= 2
        u += cT[2] * ps1; du += cT[2]; DT += cT[2]
    end
    @inbounds for k in 1:n-2
        a = (2k + 1) / (k + 1); c = k / (k + 1)
        ps2 = a * s * ps1 - c * ps0
        dp2 = dp0 + (2k + 1) * ps1
        q2 = a * s0 * q1 - c * q0
        d2 = a * (s * d1 + q1) - c * d0
        if k + 2 <= nB
            cc = cB[k+2]; b += cc * ps2; db += cc * dp2; DB += cc * d2
        end
        if k + 2 <= nT
            cc = cT[k+2]; u += cc * ps2; du += cc * dp2; DT += cc * d2
        end
        ps0, ps1 = ps1, ps2; dp0, dp1 = dp1, dp2; q0, q1 = q1, q2; d0, d1 = d1, d2
    end
    return b, db, DB, u, du, DT
end

# Apex data for polar rules: target x0 = x(s0,t0); B0 = B(s0), T0 = T(s0).
struct _Apex
    s0::Float64
    t0::Float64
    B0::Vec2
    T0::Vec2
end

"""
Line with s = sfix (Δs = sfix − s0), points t = wk.lx[1:n] with Δt = wk.ld (t − t0),
weights wk.lw (no panel Jacobian). r = x0 − y is computed from differences (no cancellation).
"""
function _sline!(wk::_AQWork{V}, P::Panel, rs::Rule1D, rt::Rule1D, kern::F, it::Int, ap::_Apex,
                 sfix::Float64, Δs::Float64, n::Int) where {F,V}
    p = wk.p; acc = wk.acc
    b, db, DB, u, du, DT = _geom_dd(P, ap.s0, Δs)
    xt = 0.5 * (u - b)
    xt0 = 0.5 * (ap.T0 - ap.B0)
    lk, lL = _line_bufs!(wk, n)
    @inbounds for m in 1:n
        t = wk.lx[m]; Δt = wk.ld[m]
        r = -(0.5 * (1 - t) * Δs * DB + 0.5 * (1 + t) * Δs * DT + Δt * xt0)
        xs = 0.5 * (1 - t) * db + 0.5 * (1 + t) * du
        J = xs[1] * xt[2] - xs[2] * xt[1]
        lk[m] = _kv(V, kern(it, r), wk.lw[m] * J)
        _lag!(wk.lt, rt, t)
        for i in 1:p
            lL[(i-1)*n+m] = wk.lt[i]
        end
    end
    _contract!(acc, lk, lL, n, p)
    wk.npts += n
    _lag!(wk.ls, rs, sfix)
    W = wk.W; ls = wk.ls
    @inbounds for j in 1:p
        aj = acc[j]; oj = (j - 1) * p
        for i in 1:p
            W[oj+i] = muladd(aj, ls[i], W[oj+i])
        end
    end
    nothing
end

"""
Line with t = tfix (Δt = tfix − t0), points s = wk.lx[1:n] with Δs = wk.ld, weights wk.lw.
"""
function _tline!(wk::_AQWork{V}, P::Panel, rs::Rule1D, rt::Rule1D, kern::F, it::Int, ap::_Apex,
                 tfix::Float64, Δt::Float64, n::Int) where {F,V}
    p = wk.p; acc = wk.acc
    a1 = 0.5 * (1 - tfix); a2 = 0.5 * (1 + tfix)
    rt0 = Δt * 0.5 * (ap.T0 - ap.B0)
    lk, lL = _line_bufs!(wk, n)
    @inbounds for m in 1:n
        s = wk.lx[m]; Δs = wk.ld[m]
        b, db, DB, u, du, DT = _geom_dd(P, ap.s0, Δs)
        r = -(a1 * Δs * DB + a2 * Δs * DT + rt0)
        xs = a1 * db + a2 * du
        xt = 0.5 * (u - b)
        J = xs[1] * xt[2] - xs[2] * xt[1]
        lk[m] = _kv(V, kern(it, r), wk.lw[m] * J)
        _lag!(wk.ls, rs, s)
        for i in 1:p
            lL[(i-1)*n+m] = wk.ls[i]
        end
    end
    _contract!(acc, lk, lL, n, p)
    wk.npts += n
    _lag!(wk.lt, rt, tfix)
    W = wk.W; lt = wk.lt
    @inbounds for j in 1:p
        lj = lt[j]; oj = (j - 1) * p
        for i in 1:p
            W[oj+i] = muladd(acc[i], lj, W[oj+i])
        end
    end
    nothing
end

# ---------------------------------------------------------------------------------------
# Adaptive 1D rule on [0,1] given complex singularities
# ---------------------------------------------------------------------------------------

# `deg`: degree of the basis along the side, which costs ⌈deg/2⌉ points on top of the kernel's estimate
function _vrule!(wk::_AQWork, opts::AreaQuadOpts, qmin::Int=opts.qmin_v, deg::Int=0)
    vx = wk.vx; vw = wk.vw; stk = wk.stk; sings = wk.sings
    empty!(vx); empty!(vw); empty!(stk)
    push!(stk, (0.0, 1.0))
    npan = 0
    shift = opts.basis_shift ? cld(deg, 2) : 0
    while !isempty(stk)
        α, β = pop!(stk)
        ρ = Inf; zw = 0.0im
        for z in sings
            r = _bernstein(z, α, β)
            if r < ρ
                ρ = r; zw = z
            end
        end
        q = ρ == Inf ? qmin : max(qmin, _qneed(ρ, opts.qtol_v) + shift)
        if q <= opts.qmax_v || npan + length(stk) >= opts.maxpanels_v || β - α < 1e-14
            q = min(q, opts.qmax_v)
            r = opts.gl[q]
            h = (β - α) / 2; c = (β + α) / 2
            for m in 1:q
                push!(vx, c + h * r.x[m]); push!(vw, h * r.w[m])
            end
            npan += 1
        else
            L = β - α
            c = clamp(real(zw), α, β)
            δ = abs(zw - c)
            if min(c - α, β - c) > 2δ
                m = c
            elseif c - α <= β - c
                m = α + max(L / 3, min(2δ, L / 2))
                m = min(m, α + L / 2)
            else
                m = β - max(L / 3, min(2δ, L / 2))
                m = max(m, β - L / 2)
            end
            push!(stk, (α, m)); push!(stk, (m, β))
        end
    end
    nothing
end

# ---------------------------------------------------------------------------------------
# Complex panel geometry (for locating near-singularities)
# ---------------------------------------------------------------------------------------

function _legendre_eval_c(c::Vector{Vec2}, s::ComplexF64)
    n = length(c)
    p0 = one(s); p1 = s
    d0 = zero(s); d1 = one(s)
    v = c[1] * p0
    dv = zero(v)
    if n >= 2
        v += c[2] * p1; dv += c[2] * d1
    end
    for k in 2:n-1
        p2 = ((2k - 1) * s * p1 - (k - 1) * p0) / k
        d2 = d0 + (2k - 1) * p1
        v += c[k+1] * p2
        dv += c[k+1] * d2
        p0, p1 = p1, p2
        d0, d1 = d1, d2
    end
    return v, dv
end

const _PLAIN = 0
const _TAU = 1
const _BOTTOM = 2

# g(v) = (x − x0)₁ + i (x − x0)₂ along the slice y = A + u (P1 + v (P2−P1) − A), and g'(v)
function _slice_g(P::Panel, mode::Int, A::Vec2, P1::Vec2, P2::Vec2, u::Float64, x0::Vec2, v::ComplexF64)
    bs = P1[1] + v * (P2[1] - P1[1]); bσ = P1[2] + v * (P2[2] - P1[2])
    s = A[1] + u * (bs - A[1]); σ = A[2] + u * (bσ - A[2])
    ds = u * (P2[1] - P1[1]); dσ = u * (P2[2] - P1[2])
    if mode == _TAU
        t = -1 + 2σ^2; dt = 4σ * dσ
    else
        t = σ; dt = dσ
    end
    B, dB = _legendre_eval_c(P.B, s)
    T, dT = _legendre_eval_c(P.T, s)
    x = 0.5 * (1 - t) * B + 0.5 * (1 + t) * T
    xs = 0.5 * (1 - t) * dB + 0.5 * (1 + t) * dT
    xt = 0.5 * (T - B)
    c1 = x[1] - x0[1]; c2 = x[2] - x0[2]
    d = xs * ds + xt * dt
    return c1 + im * c2, d[1] + im * d[2], abs2(c1) + abs2(c2)
end

"Complex root (near [0,1]) of the slice distance function, or NaN."
function _slice_root(P::Panel, mode::Int, A::Vec2, P1::Vec2, P2::Vec2, u::Float64, x0::Vec2)
    vbest = 0.0; dbest = Inf
    for k in 0:10
        v = k / 10
        _, _, d2 = _slice_g(P, mode, A, P1, P2, u, x0, complex(v))
        if d2 < dbest
            dbest = d2; vbest = v
        end
    end
    v = complex(vbest)
    for _ in 1:12
        g, dg, _ = _slice_g(P, mode, A, P1, P2, u, x0, v)
        dg == 0 && return complex(NaN)
        Δ = g / dg
        v -= Δ
        abs(v) > 20 && return complex(NaN)
        abs(Δ) < 1e-9 * (1 + abs(v)) && break
    end
    return v
end

# ---------------------------------------------------------------------------------------
# Polar (Duffy) rule for targets on the panel
# ---------------------------------------------------------------------------------------

"""
Accumulate into wk.W the weights for a target x0 = x(s0,t0) lying on panel P (closed square).
"""
function _polar!(wk::_AQWork, P::Panel, rs::Rule1D, rt::Rule1D, kern::F, it::Int, x0::Vec2,
                 s0::Float64, t0::Float64, opts::AreaQuadOpts) where {F}
    if !P.weighted
        mode = _PLAIN; A = Vec2(s0, t0); σlo = -1.0
    elseif t0 <= -1.0
        mode = _BOTTOM; A = Vec2(s0, -1.0); σlo = -1.0
    else
        mode = _TAU; A = Vec2(s0, sqrt((1 + t0) / 2)); σlo = 0.0
    end
    σhi = 1.0
    _, xs0, xt0, _ = panel_map(P, s0, max(t0, -1.0))
    ap = _Apex(s0, mode == _BOTTOM ? -1.0 : t0, legendre_eval(P.B, s0)[1], legendre_eval(P.T, s0)[1])
    X = mode == _TAU ? hcat(xs0, 4 * A[2] * xt0) : hcat(xs0, xt0)
    X = SMatrix{2,2,Float64,4}(X)
    # sides: (P1, P2, vertical?)
    sides = ((Vec2(-1.0, σlo), Vec2(1.0, σlo), false),   # bottom
             (Vec2(1.0, σlo), Vec2(1.0, σhi), true),     # right
             (Vec2(-1.0, σhi), Vec2(1.0, σhi), false),   # top
             (Vec2(-1.0, σlo), Vec2(-1.0, σhi), true))   # left
    for (P1, P2, vert) in sides
        h = vert ? abs(P1[1] - A[1]) : abs(P1[2] - A[2])
        h <= 1e-14 && continue
        detA = h * norm(P2 - P1)
        zsub = (mode == _BOTTOM) && vert          # v = z², weight v^{-1/2}
        # --- singularities along the base
        sings = wk.sings; empty!(sings)
        e = X * (P1 - A); f = X * (P2 - P1)
        ff = dot(f, f)
        v0 = complex(-dot(e, f) / ff, abs(e[1] * f[2] - e[2] * f[1]) / ff)
        push!(sings, v0)
        for us in opts.slice_us
            vr = _slice_root(P, mode, A, P1, P2, us, x0)
            if isfinite(vr)
                push!(sings, vr)
                if mode == _TAU && vert
                    # t depends on σ²: mirror image σ → −σ of the root
                    σr = A[2] + us * (vr - A[2])
                    push!(sings, (-σr - A[2]) / us + A[2])
                end
            end
        end
        if zsub
            nz = length(sings)
            for k in 1:nz
                z = sqrt(sings[k])
                sings[k] = z
                push!(sings, -z)
            end
        end
        dbl = mode != _PLAIN && vert                  # the side runs in σ or z: basis degree 2(p − 1)
        _vrule!(wk, opts, dbl ? opts.qmin_v2 : opts.qmin_v, dbl ? 2(wk.p - 1) : wk.p - 1)
        nv = length(wk.vx)
        # --- radial lines
        ur = mode == _BOTTOM ? opts.uhalf : mode == _TAU ? opts.utau : opts.ulog
        fac = mode == _TAU ? 2sqrt(2.0) : (mode == _BOTTOM ? 1 / sqrt(2.0) : 1.0)
        resize!(wk.lx, nv); resize!(wk.lw, nv); resize!(wk.ld, nv)
        for (u, wu) in zip(ur.x, ur.w)
            wbase = wu * u * detA * fac
            if vert
                Δs = u * (P1[1] - A[1])
                sfix = A[1] + Δs
                for m in 1:nv
                    v = wk.vx[m]; wv = wk.vw[m]
                    if zsub
                        wv *= 2; v = v^2
                    end
                    Δσ = u * (P1[2] + v * (P2[2] - P1[2]) - A[2])
                    σ = A[2] + Δσ
                    if mode == _TAU
                        wk.lx[m] = -1 + 2σ^2; wk.ld[m] = 2Δσ * (σ + A[2])
                    else
                        wk.lx[m] = σ; wk.ld[m] = Δσ
                    end
                    wk.lw[m] = wbase * wv
                end
                _sline!(wk, P, rs, rt, kern, it, ap, sfix, Δs, nv)
            else
                Δσ = u * (P1[2] - A[2])
                σfix = A[2] + Δσ
                if mode == _TAU
                    tfix = -1 + 2σfix^2; Δt = 2Δσ * (σfix + A[2])
                else
                    tfix = σfix; Δt = Δσ
                end
                for m in 1:nv
                    v = wk.vx[m]
                    Δs = u * (P1[1] + v * (P2[1] - P1[1]) - A[1])
                    wk.lx[m] = A[1] + Δs; wk.ld[m] = Δs
                    wk.lw[m] = wbase * wk.vw[m]
                end
                _tline!(wk, P, rs, rt, kern, it, ap, tfix, Δt, nv)
            end
        end
    end
    nothing
end

# ---------------------------------------------------------------------------------------
# Adaptive tensor boxes for targets off the panel
# ---------------------------------------------------------------------------------------

function _box_tensor!(wk::_AQWork{V}, P::Panel, rs::Rule1D, rt::Rule1D, kern::F, it::Int, x0::Vec2,
                      s1, s2, t1, t2, q::Int, opts::AreaQuadOpts) where {F,V}
    p = wk.p
    r = opts.gl[q]
    hs = (s2 - s1) / 2; cs = (s2 + s1) / 2
    LS = wk.LS
    @inbounds for a in 1:q
        s = cs + hs * r.x[a]
        b, db = legendre_eval(P.B, s)
        u, du = legendre_eval(P.T, s)
        wk.bB[a] = (b, db, u, du)
        wk.bsw[a] = hs * r.w[a]
        _lag!(wk.ls, rs, s)
        for i in 1:p
            LS[(i-1)*q+a] = wk.ls[i]
        end
    end
    gjb = P.weighted && t1 <= -1.0
    rtq = gjb ? opts.gj[q] : r
    ht = (t2 - t1) / 2; ct = (t2 + t1) / 2
    W = wk.W; kb = wk.kb; tb = wk.tb; LT = wk.LT
    @inbounds for b in 1:q
        y = rtq.x[b]
        t = ct + ht * y
        wt = gjb ? sqrt(ht) * rtq.w[b] : ht * rtq.w[b]
        if P.weighted && !gjb
            wt /= sqrt(1 + t)
        end
        a1 = 0.5 * (1 - t); a2 = 0.5 * (1 + t)
        for a in 1:q
            bb, db, uu, du = wk.bB[a]
            x = a1 * bb + a2 * uu
            xs = a1 * db + a2 * du
            xt = 0.5 * (uu - bb)
            J = xs[1] * xt[2] - xs[2] * xt[1]
            kb[a] = _kv(V, kern(it, x0 - x), wk.bsw[a] * wt * J)
        end
        # contract over s: accumulators in registers, two coefficients and two partial sums
        # at a time (independent FMA chains)
        i = 1
        while i < p
            oi = (i - 1) * q; oj = oi + q
            c1 = zero(V); c2 = c1; d1 = c1; d2 = c1
            a = 1
            while a < q
                k1 = kb[a]; k2 = kb[a+1]
                c1 = muladd(k1, LS[oi+a], c1); d1 = muladd(k2, LS[oi+a+1], d1)
                c2 = muladd(k1, LS[oj+a], c2); d2 = muladd(k2, LS[oj+a+1], d2)
                a += 2
            end
            if a == q
                c1 = muladd(kb[a], LS[oi+a], c1); c2 = muladd(kb[a], LS[oj+a], c2)
            end
            ob = (b - 1) * p
            tb[ob+i] = c1 + d1; tb[ob+i+1] = c2 + d2
            i += 2
        end
        if i == p
            c1 = zero(V); oi = (i - 1) * q
            for a in 1:q
                c1 = muladd(kb[a], LS[oi+a], c1)
            end
            tb[(b-1)*p+i] = c1
        end
        _lag!(wk.lt, rt, t)
        for j in 1:p
            LT[(j-1)*q+b] = wk.lt[j]
        end
    end
    # contract over t: W[i,j] += Σ_b tb[i,b] LT[b,j]  (registers, two independent chains)
    @inbounds for j in 1:p
        oj = (j - 1) * q; ow = (j - 1) * p
        for i in 1:p
            c1 = zero(V); d1 = c1
            b = 1
            while b < q
                c1 = muladd(tb[(b-1)*p+i], LT[oj+b], c1)
                d1 = muladd(tb[b*p+i], LT[oj+b+1], d1)
                b += 2
            end
            b == q && (c1 = muladd(tb[(b-1)*p+i], LT[oj+b], c1))
            W[ow+i] += c1 + d1
        end
    end
    wk.npts += q * q
    nothing
end

function _boxes!(wk::_AQWork, P::Panel, rs::Rule1D, rt::Rule1D, kern::F, it::Int, x0::Vec2,
                 opts::AreaQuadOpts) where {F}
    stk = wk.bstk; empty!(stk)
    push!(stk, (-1.0, 1.0, -1.0, 1.0, 0.0))
    while !isempty(stk)
        s1, s2, t1, t2, dep = pop!(stk)
        sc = (s1 + s2) / 2; tc = (t1 + t2) / 2
        xc = panel_map(P, sc, tc)[1]
        R = 0.0
        for (s, t) in ((s1, t1), (s2, t1), (s2, t2), (s1, t2), (sc, t1), (s2, tc), (sc, t2), (s1, tc))
            R = max(R, norm(panel_map(P, s, t)[1] - xc))
        end
        dist = norm(x0 - xc)
        q = opts.qmin_box
        ok = dist > 1.02R
        if ok
            xr = dist / R - 1
            q = max(q, _qneed(xr + sqrt(xr^2 + 1), opts.qtol))
        end
        if P.weighted && t1 > -1.0
            xw = (tc + 1) / ((t2 - t1) / 2)
            q = max(q, _qneed(xw + sqrt(xw^2 - 1), opts.qtol))
        end
        if (ok && q <= opts.qmax_box) || dep >= opts.maxdepth
            _box_tensor!(wk, P, rs, rt, kern, it, x0, s1, s2, t1, t2, min(q, opts.qmax_box), opts)
        else
            ws = norm(panel_map(P, s2, tc)[1] - panel_map(P, s1, tc)[1])
            ht = norm(panel_map(P, sc, t2)[1] - panel_map(P, sc, t1)[1])
            if ws > 2ht
                push!(stk, (s1, sc, t1, t2, dep + 1)); push!(stk, (sc, s2, t1, t2, dep + 1))
            elseif ht > 2ws
                push!(stk, (s1, s2, t1, tc, dep + 1)); push!(stk, (s1, s2, tc, t2, dep + 1))
            else
                push!(stk, (s1, sc, t1, tc, dep + 1)); push!(stk, (sc, s2, t1, tc, dep + 1))
                push!(stk, (s1, sc, tc, t2, dep + 1)); push!(stk, (sc, s2, tc, t2, dep + 1))
            end
        end
    end
    nothing
end

# ---------------------------------------------------------------------------------------
# Parameter preimage of a point (Newton)
# ---------------------------------------------------------------------------------------

"""
    panel_preimage(P, x0) -> (s, t, resid)

Newton solve of x(s,t) = x0 (iterates are kept in [-2,2]²). `resid` = |x(s,t) − x0|.
"""
function panel_preimage(P::Panel, x0::Vec2)
    s = 0.0; t = 0.0; best = Inf
    for sg in (-0.75, 0.0, 0.75), tg in (-0.75, 0.0, 0.75)
        d = norm(panel_map(P, sg, tg)[1] - x0)
        if d < best
            best = d; s = sg; t = tg
        end
    end
    res = Inf
    for _ in 1:40
        x, xs, xt, J = panel_map(P, s, t)
        r = x0 - x
        res = norm(r)
        abs(J) < 1e-300 && break
        Δs = (r[1] * xt[2] - r[2] * xt[1]) / J
        Δt = (xs[1] * r[2] - xs[2] * r[1]) / J
        s = clamp(s + Δs, -2.0, 2.0); t = clamp(t + Δt, -2.0, 2.0)
        abs(Δs) + abs(Δt) < 1e-15 && (res = norm(x0 - panel_map(P, s, t)[1]); break)
    end
    return s, t, res
end

"""
Return (on, s0, t0): whether x0 lies on panel P (closed square, up to `rtol`·diam), with
parameters snapped to the sides.
"""
function _on_panel(P::Panel, x0::Vec2, diam::Float64; rtol=1e-11)
    s, t, res = panel_preimage(P, x0)
    ptol = 1e-10
    if res <= rtol * diam && abs(s) <= 1 + ptol && abs(t) <= 1 + ptol
        abs(s - 1) <= ptol && (s = 1.0); abs(s + 1) <= ptol && (s = -1.0)
        abs(t - 1) <= ptol && (t = 1.0); abs(t + 1) <= ptol && (t = -1.0)
        return true, clamp(s, -1.0, 1.0), clamp(t, -1.0, 1.0)
    end
    return false, s, t
end

# ---------------------------------------------------------------------------------------
# Single (target, panel) local weights
# ---------------------------------------------------------------------------------------

_wrap_kernel(K) = hasmethod(K, Tuple{Int,Vec2}) ? K : ((i, r) -> K(r))

"""
Compute into wk.W the local weights of panel k for target x0. `on` = (s0,t0) if the target
lies on the panel (closed square), otherwise `nothing` (adaptive boxes).
"""
function _local_weights!(wk::_AQWork{V}, ad::AreaDisc, k::Int, kern::F, it::Int, x0::Vec2, on,
                         opts::AreaQuadOpts, P::Panel=ad.panels[k]) where {F,V}
    fill!(wk.W, zero(V))
    rt = trule(ad, k)
    if on === nothing
        _boxes!(wk, P, ad.rs, rt, kern, it, x0, opts)
    else
        _polar!(wk, P, ad.rs, rt, kern, it, x0, Float64(on[1]), Float64(on[2]), opts)
    end
    wk.W
end

"""
    area_local_weights(ad, k, x0, K; on=:auto, tol=DEFAULT_TOL, opts=nothing, itarg=0)

Accurate local weights of panel `k` for the target `x0`: a vector of p² 2×2 blocks W[m]
(node order of `panel_nodes(ad,k)`) such that ∫_panel K(x0 − y) μ(y) dA ≈ Σ_m W[m] μ̃_m.
`K` is K(r) or K(i, r) (then called with i = `itarg`). `on`: `(s0,t0)` if x0 = x(s0,t0) lies
on the panel, `nothing` if it does not, `:auto` to detect it.
"""
function area_local_weights(ad::AreaDisc, k::Int, x0::Vec2, K; on=:auto, tol=DEFAULT_TOL,
                            opts=nothing, itarg::Int=0)
    opts = opts === nothing ? AreaQuadOpts(ad.p; tol=tol) : opts
    kern = _wrap_kernel(K)
    if on === :auto
        fl, s0, t0 = _on_panel(ad.panels[k], x0, panel_diameter(ad.panels[k]))
        on = fl ? (s0, t0) : nothing
    end
    wk = _AQWork(ad.p, max(opts.qmax_box, opts.qmax_v))
    W = _local_weights!(wk, ad, k, kern, itarg, x0, on, opts, _trim_panel(ad.panels[k]))
    p = ad.p
    [SMatrix{2,2,Float64,4}(Tuple(W[i, j])) for j in 1:p for i in 1:p]
end

# ---------------------------------------------------------------------------------------
# Spatial lookup of near panels
# ---------------------------------------------------------------------------------------

struct _PanelGrid
    lo::Vec2
    h::Float64
    nx::Int
    ny::Int
    cells::Vector{Vector{Int32}}
end

function _PanelGrid(cen::Vector{Vec2}, rad::Vector{Float64})
    n = length(cen)
    lo = Vec2(minimum(c[1] - r for (c, r) in zip(cen, rad)), minimum(c[2] - r for (c, r) in zip(cen, rad)))
    hi = Vec2(maximum(c[1] + r for (c, r) in zip(cen, rad)), maximum(c[2] + r for (c, r) in zip(cen, rad)))
    h = max(sort(rad)[cld(n, 2)], 1e-300) * 1.0
    span = hi - lo
    nx = clamp(ceil(Int, span[1] / h), 1, 4096); ny = clamp(ceil(Int, span[2] / h), 1, 4096)
    h = max(span[1] / nx, span[2] / ny) * (1 + 1e-12)
    cells = [Int32[] for _ in 1:nx*ny]
    for k in 1:n
        c = cen[k]; r = rad[k]
        i1 = clamp(floor(Int, (c[1] - r - lo[1]) / h) + 1, 1, nx); i2 = clamp(floor(Int, (c[1] + r - lo[1]) / h) + 1, 1, nx)
        j1 = clamp(floor(Int, (c[2] - r - lo[2]) / h) + 1, 1, ny); j2 = clamp(floor(Int, (c[2] + r - lo[2]) / h) + 1, 1, ny)
        for j in j1:j2, i in i1:i2
            push!(cells[i+(j-1)*nx], Int32(k))
        end
    end
    _PanelGrid(lo, h, nx, ny, cells)
end

function _near_panels!(out::Vector{Int32}, g::_PanelGrid, x::Vec2, cen, rad)
    empty!(out)
    i = floor(Int, (x[1] - g.lo[1]) / g.h) + 1
    j = floor(Int, (x[2] - g.lo[2]) / g.h) + 1
    (1 <= i <= g.nx && 1 <= j <= g.ny) || return out
    for k in g.cells[i+(j-1)*g.nx]
        norm(x - cen[k]) < rad[k] && push!(out, k)
    end
    sort!(out)
    out
end

# ---------------------------------------------------------------------------------------
# Boundary (Γ) targets
# ---------------------------------------------------------------------------------------

"""
    boundary_target_info(ad, bd) -> Vector{Tuple{Int,Vec2}}

For each node of the boundary discretization `bd`, the edge panel k whose bottom edge
(t = −1) contains it and its parameters (s0, −1). Entry (0, ·) if none is found.
Use as `area_correction(ad, K; targets=bd.x, target_info=boundary_target_info(ad, bd))`.
"""
function boundary_target_info(ad::AreaDisc, bd::BoundaryDisc)
    info = Vector{Tuple{Int,Vec2}}(undef, length(bd.x))
    byb = Dict{Int,Vector{Int}}()
    for (k, P) in enumerate(ad.panels)
        P.weighted && P.bpanel > 0 && push!(get!(byb, P.bpanel, Int[]), k)
    end
    for b in 1:npanels(bd), (i, s) in enumerate(bd.rule.x)
        idx = (b - 1) * bd.p + i
        x0 = bd.x[idx]
        info[idx] = (0, Vec2(0.0, -1.0))
        for k in get(byb, b, Int[])
            P = ad.panels[k]
            # bottom curve B(s): 1D Newton from the boundary parameter
            ss = s
            for _ in 1:30
                xb, dxb = legendre_eval(P.B, ss)
                Δ = dot(dxb, x0 - xb) / dot(dxb, dxb)
                ss += Δ
                abs(Δ) < 1e-15 && break
            end
            xb = legendre_eval(P.B, ss)[1]
            if abs(ss) <= 1 + 1e-12 && norm(xb - x0) <= 1e-11 * panel_diameter(P)
                info[idx] = (k, Vec2(clamp(ss, -1.0, 1.0), -1.0))
                break
            end
        end
    end
    info
end

# ---------------------------------------------------------------------------------------
# Assembly of the correction matrix
# ---------------------------------------------------------------------------------------

"""
    AreaUpsampling(ad, q)

Upsampled tensor rule on every panel (q×q; Gauss–Legendre in s, Gauss–Legendre or
Gauss–Jacobi (weighted panels) in t). Densities are interpolated from the p×p nodes with the
reference matrices `Iu`/`Iw` (q² × p², s index fastest), so the far field can be evaluated from
the points `x` with weights `w` (Jacobi weight and Jacobian included). With a q-point rule the
smooth quadrature is accurate closer to a panel, so fewer (target, panel) pairs need corrections.
"""
struct AreaUpsampling
    q::Int
    x::Vector{Vec2}
    w::Vector{Float64}
    Iu::Matrix{Float64}
    Iw::Matrix{Float64}
end

function AreaUpsampling(ad::AreaDisc, q::Int)
    rs = gl_rule(q); rt = gl_rule(q); rtw = gj_rule(q)
    Is = interp_matrix(ad.rs, rs.x)
    Iu = kron(interp_matrix(ad.rt, rt.x), Is)
    Iw = kron(interp_matrix(ad.rtw, rtw.x), Is)
    q2 = q * q
    x = Vector{Vec2}(undef, npanels(ad) * q2); w = zeros(npanels(ad) * q2)
    Threads.@threads for k in 1:npanels(ad)
        P = ad.panels[k]
        tr = P.weighted ? rtw : rt
        for j in 1:q, i in 1:q
            idx = (k - 1) * q2 + i + (j - 1) * q
            xx, _, _, J = panel_map(P, rs.x[i], tr.x[j])
            x[idx] = xx
            w[idx] = rs.w[i] * tr.w[j] * J
        end
    end
    AreaUpsampling(q, x, w, Iu, Iw)
end

"""Interpolate nodal values (stored μ̃ format) to the upsampled points."""
function upsample(up::AreaUpsampling, ad::AreaDisc, v::AbstractVector{T}) where {T}
    p2 = ad.p^2; q2 = up.q^2
    out = Vector{T}(undef, npanels(ad) * q2)
    Threads.@threads for k in 1:npanels(ad)
        I_ = ad.panels[k].weighted ? up.Iw : up.Iu
        src = (k - 1) * p2
        dst = (k - 1) * q2
        for a in 1:q2
            acc = zero(T)
            @inbounds for b in 1:p2
                acc += I_[a, b] * v[src+b]
            end
            out[dst+a] = acc
        end
    end
    out
end

"""
    area_correction(ad, K; targets=nothing, target_info=nothing, tol=DEFAULT_TOL,
                    near_factor=nothing, opts=nothing, detect_on_panel=true,
                    transposed=false, stats=nothing, upsampling=nothing, keep=nothing,
                    ncomp=nothing) -> SparseMatrixCSC  (or a Vector of them, with `ncomp`)

Sparse correction C (2Nt × 2N, interleaved) such that, for the area potential
u(xᵢ) = ∫_Ω K(xᵢ − y) μ(y) dA_y,

    u(xᵢ) ≈ Σ_{j ≠ i} K(xᵢ − x_j) w_j μ̃_j + (C μ̃)ᵢ          (smooth rule as in the FMM).

* `targets === nothing`: targets are the area nodes; the smooth-rule self term (j = i) is
  excluded (it is 0 in the FMM).
* `targets::Vector{Vec2}`: arbitrary targets; the smooth rule includes all j (except
  sources coinciding exactly with the target). `target_info[i] = (k, (s0,t0))` declares that
  target i lies on panel k at parameters (s0,t0) (e.g. Γ nodes: `boundary_target_info`);
  k = 0 means no information. With `detect_on_panel=true` targets lying on other near panels
  (shared sides/corners) are detected automatically by a Newton solve.
* `K`: K(r)::SMatrix{2,2} or the target-dependent K(i, r) (i = target index), e.g. a
  traction kernel with the normal of target i.
* `ncomp = M`: K returns M scalar components instead (an `SVector{M}`, a tuple of length M, or a
  number for M = 1), and the result is a vector of M scalar corrections Cᶜ (Nt × N, one row per
  target, one column per source node), all from one pass: the rules depend on the geometry only.
  They share one sparsity pattern (the same `colptr` and `rowval` arrays). Each equals the
  correction of the scalar kernel Kᶜ alone (bit for bit, up to the fused multiply–adds the compiler
  may choose for a different M).
* `transposed=true` returns Cᵀ (2N × 2Nt; with `ncomp`, N × Nt each), which is what is
  assembled internally (saves a copy).
* `stats`: optional Dict to receive statistics (pairs, kernel evaluations, nnz).
* `upsampling::AreaUpsampling`: the far field is evaluated from the upsampled rule instead of
  the nodes; the corrections subtract the upsampled smooth weights and the near radius shrinks.
* `keep(i, k)::Bool`: correct only the near (target, panel) pairs it accepts; the others keep
  the smooth rule (e.g. only pairs from different drops).
"""
function area_correction(ad::AreaDisc, K; targets=nothing, target_info=nothing, tol::Real=DEFAULT_TOL,
                         near_factor=nothing, opts=nothing, detect_on_panel::Bool=true,
                         transposed::Bool=false, stats=nothing, upsampling=nothing, keep=nothing,
                         ncomp::Union{Nothing,Integer}=nothing)
    if near_factor === nothing && upsampling !== nothing
        # smooth q-point rule: needed distance from the Bernstein-ellipse estimate with q
        ρ = (1 / tol)^(1 / (2upsampling.q))
        near_factor = _near_floor((1 + (ρ - 1 / ρ) / 2) / sqrt(2))
    end
    # (a new binding: reassigning the keyword would box it in the threaded closures below)
    qo = opts === nothing ? AreaQuadOpts(ad.p; tol=tol, near_factor=near_factor) : opts
    kern = _wrap_kernel(K)
    selfmode = targets === nothing
    X = selfmode ? ad.x : collect(Vec2, targets)
    geo = _near_setup(ad, X, qo, selfmode, target_info, keep)
    if ncomp === nothing
        return _correction_pass(Val(0), ad, kern, X, geo, qo, selfmode, target_info, detect_on_panel, upsampling,
                                transposed, stats)
    end
    ncomp >= 1 || throw(ArgumentError("area_correction: ncomp ≥ 1"))
    _correction_pass(Val(Int(ncomp)), ad, _Components{Int(ncomp),typeof(kern)}(kern), X, geo, qo, selfmode,
                     target_info, detect_on_panel, upsampling, transposed, stats)
end

# panel geometry (center, circumradius, diameter, trimmed copies) and the near panels of every target
function _near_setup(ad::AreaDisc, X::Vector{Vec2}, qo::AreaQuadOpts, selfmode::Bool, target_info, keep)
    p2 = ad.p^2; np = npanels(ad); Nt = length(X)
    cen = Vector{Vec2}(undef, np); crad = zeros(np); diam = zeros(np)
    for (k, P) in enumerate(ad.panels)
        c = panel_center(P); cen[k] = c
        r = 0.0
        for m in 0:15
            τ = -1 + m / 8
            for (s, t) in ((τ, -1.0), (1.0, τ), (-τ, 1.0), (-1.0, -τ))
                r = max(r, norm(panel_map(P, s, t)[1] - c))
            end
        end
        crad[k] = r; diam[k] = panel_diameter(P)
    end
    nrad = qo.near_factor .* crad
    tpan = [_trim_panel(P) for P in ad.panels]
    grid = _PanelGrid(cen, nrad)
    near = Vector{Vector{Int32}}(undef, Nt)
    Threads.@threads for i in 1:Nt
        out = Int32[]
        _near_panels!(out, grid, X[i], cen, nrad)
        if selfmode
            k0 = (i - 1) ÷ p2 + 1
            insorted(k0, out) || (push!(out, Int32(k0)); sort!(out))
        elseif target_info !== nothing && target_info[i][1] > 0
            k0 = target_info[i][1]
            insorted(k0, out) || (push!(out, Int32(k0)); sort!(out))
        end
        keep === nothing || filter!(k -> keep(i, Int(k)), out)
        near[i] = out
    end
    (; cen, crad, diam, tpan, near)
end

# Local weights of the pair (target i, panel k) minus the smooth rule's (the nodes, or the upsampled points), in wk.W
function _pair_weights!(wk::_AQWork{V}, ad::AreaDisc, kern::F, i::Int, x0::Vec2, k::Int, geo, qo, selfmode::Bool,
                        target_info, detect_on_panel::Bool, upsampling) where {V,F}
    p = ad.p; p2 = p * p
    P = ad.panels[k]
    on = nothing
    jself = 0
    if selfmode
        if (i - 1) ÷ p2 + 1 == k
            on = (ad.st[i][1], ad.st[i][2]); jself = i
        end
    else
        if target_info !== nothing && target_info[i][1] == k
            on = (target_info[i][2][1], target_info[i][2][2])
        elseif detect_on_panel && norm(x0 - geo.cen[k]) <= 1.05geo.crad[k]
            fl, s0, t0 = _on_panel(P, x0, geo.diam[k])
            fl && (on = (s0, t0))
        end
    end
    W = _local_weights!(wk, ad, k, kern, i, x0, on, qo, geo.tpan[k])
    base = (k - 1) * p2
    if upsampling !== nothing
        # subtract the upsampled smooth rule: Σ_u K(x0 − y_u) ω_u ℓ_l(y_u)
        q2 = upsampling.q^2
        I_ = P.weighted ? upsampling.Iw : upsampling.Iu
        ub = (k - 1) * q2
        for u in 1:q2
            r = x0 - upsampling.x[ub+u]
            (r[1] == 0 && r[2] == 0) && continue
            Ks = _kv(V, kern(i, r), upsampling.w[ub+u])
            for jj in 1:p, ii in 1:p
                c = I_[u, ii + (jj - 1) * p]
                W[ii, jj] -= c * Ks
            end
        end
    else
        for jj in 1:p, ii in 1:p
            j = base + ii + (jj - 1) * p
            j == jself && continue
            r = x0 - ad.x[j]
            if !(selfmode && j == i) && (r[1] != 0 || r[2] != 0)
                W[ii, jj] -= _kv(V, kern(i, r), ad.w[j])
            end
        end
    end
    W
end

_acc_type(::Val{0}) = SVector{4,Float64}
_acc_type(::Val{M}) where {M} = SVector{M,Float64}

# pass 2: the weights of every pair. Val(0): the 2×2 form, interleaved (2N × 2Nt); Val(M): M scalar corrections
# (N × Nt each) on one pattern
function _correction_pass(::Val{M}, ad::AreaDisc, kern::F, X, geo, qo, selfmode, target_info, detect_on_panel,
                          upsampling, transposed, stats) where {M,F}
    p = ad.p; p2 = p * p; N = length(ad); Nt = length(X)
    near = geo.near
    nb = M == 0 ? 2 : 1                            # columns per target, entries per node in a column
    colptr = Vector{Int}(undef, nb * Nt + 1)
    colptr[1] = 1
    for i in 1:Nt, c in 1:nb
        colptr[nb*(i-1)+c+1] = colptr[nb*(i-1)+c] + nb * p2 * length(near[i])
    end
    nnzt = colptr[end] - 1
    rowval = Vector{Int}(undef, nnzt)
    nzval = [Vector{Float64}(undef, nnzt) for _ in 1:max(M, 1)]
    nthr = Threads.nthreads()
    nextt = Threads.Atomic{Int}(1)
    npts = Threads.Atomic{Int}(0)
    qmx = max(qo.qmax_box, qo.qmax_v)
    Threads.@threads for _ in 1:nthr
        wk = _AQWork{_acc_type(Val(M))}(p, qmx)
        while true
            i = Threads.atomic_add!(nextt, 1)
            i > Nt && break
            x0 = X[i]
            for (m, k) in enumerate(near[i])
                k = Int(k)
                W = _pair_weights!(wk, ad, kern, i, x0, k, geo, qo, selfmode, target_info, detect_on_panel, upsampling)
                base = (k - 1) * p2
                if M == 0
                    pos1 = colptr[2i-1]; pos2 = colptr[2i]; nz = nzval[1]
                    for jj in 1:p, ii in 1:p
                        l = ii + (jj - 1) * p
                        j = base + l
                        w11, w21, w12, w22 = W[ii, jj]
                        off = ((m - 1) * p2 + (l - 1)) * 2
                        # column 2i-1 (row component 1 of C): entries (2j-1, 2j) ← W[1,1], W[1,2]
                        rowval[pos1+off] = 2j - 1; nz[pos1+off] = w11
                        rowval[pos1+off+1] = 2j; nz[pos1+off+1] = w12
                        rowval[pos2+off] = 2j - 1; nz[pos2+off] = w21
                        rowval[pos2+off+1] = 2j; nz[pos2+off+1] = w22
                    end
                else
                    pos = colptr[i] + (m - 1) * p2
                    for jj in 1:p, ii in 1:p
                        l = ii + (jj - 1) * p
                        rowval[pos+l-1] = base + l
                        w = W[ii, jj]
                        for c in 1:M
                            nzval[c][pos+l-1] = w[c]
                        end
                    end
                end
            end
        end
        Threads.atomic_add!(npts, wk.npts)
    end
    if stats !== nothing
        stats[:pairs] = sum(length, near)
        stats[:kernel_evals] = npts[]
        stats[:nnz] = nnzt
    end
    if M == 0
        Ct = SparseMatrixCSC(2N, 2Nt, colptr, rowval, nzval[1])
        return transposed ? Ct : copy(transpose(Ct))
    end
    Cts = [SparseMatrixCSC(N, Nt, colptr, rowval, nzval[c]) for c in 1:M]
    transposed && return Cts
    # Cᶜ = (Cᶜᵀ)ᵀ: one transpose of the pattern (a permutation), applied to every component
    perm = SparseMatrixCSC(N, Nt, colptr, rowval, collect(1.0:nnzt))
    Pt = copy(transpose(perm))
    idx = round.(Int, Pt.nzval)
    [SparseMatrixCSC(Nt, N, Pt.colptr, Pt.rowval, nzval[c][idx]) for c in 1:M]
end
