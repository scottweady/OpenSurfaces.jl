# Local quadrature for layer potentials on the boundary curve Γ (a BoundaryDisc).
#
#   S²ᴰ_Γ[ν](x) = ∮_Γ G2(x−y) ν(y) ds_y,   G2(r) = (1/(4π μ_2D)) (−log ρ I + r rᵀ/ρ²)
#
# Pieces:
#   * Self panel (target = a node of the panel): the log singularity is split off,
#         −log|r(s)| = −log|sᵢ−s| − log(|r(s)|/|sᵢ−s|),
#     the first part is product-integrated exactly against Legendre polynomials on an
#     upsampled qₛ-point Gauss–Legendre grid (analytic log moments via Legendre Q functions),
#     the second part and r rᵀ/ρ² are smooth in s and use plain qₛ-point Gauss–Legendre.
#   * Every other near (target, panel) pair: adaptive bisection in the panel parameter s with
#     q-point Gauss–Legendre leaves. A leaf [a,b] is accepted when
#         |x − γ(m)| ≥ α |γ'(m)| (b−a),   m = (a+b)/2,
#     so the nearest complex singularity of the integrand stays outside the Bernstein ellipse
#     with ρ ≈ 3.7 (error ~ ρ^{-2q}). This part is kernel-agnostic (any K(r)::Mat2).
#   * Near lists: a panel k is "near" a target x if |x − cₖ| < rnearₖ = β Lₖ (cₖ = γₖ(0),
#     Lₖ = arc length), with β chosen from the p-point Gauss–Legendre convergence rate and `tol`.
#
# Corrections: C = accurate weights − smooth-rule weights in the interleaved
# 2×2 layout. For Γ targets the smooth weight of the target node itself is 0 (self excluded).
#
# Traction on Γ (interior limit, outward normal n):
#   T_ΓΓ ν(x) = ½ ν(x) + PV∮ T2(x−y, n(x)) ν(y) ds_y,   T2(r,n) f = −(1/π) r (r·n)(r·f)/ρ⁴.
#   The jump constant +½ follows from ∇·σ = −ν δ_Γ (the single layer is a force density ν
#   acting on the fluid): t_int − t_ext = ν, PV = ½(t_int + t_ext).
#   The PV kernel is smooth along Γ, with diagonal limit
#       lim_{y→x} T2(x−y, n(x)) = −(κ(x)/(2π)) τ τᵀ,
#   so the plain Gauss rule with this diagonal value is spectrally accurate.

using SparseArrays

# ---------------------------------------------------------------------------------------
# Local point kernels (private copies, so this file does not depend on kernels.jl)
# ---------------------------------------------------------------------------------------

@inline function _lq_G2(r::Vec2, μ_2D::Real)
    ρ2 = r[1]^2 + r[2]^2
    c = 1 / (4π * μ_2D)
    l = -0.5 * log(ρ2)
    a = r[1]^2 / ρ2; b = r[1] * r[2] / ρ2; d = r[2]^2 / ρ2
    Mat2(c * (l + a), c * b, c * b, c * (l + d))
end

"Traction kernel at x with normal n for a point force at y, r = x − y: t = T2 f."
@inline function _lq_T2(r::Vec2, n::Vec2)
    ρ2 = r[1]^2 + r[2]^2
    s = -(r[1] * n[1] + r[2] * n[2]) / (π * ρ2^2)
    Mat2(s * r[1] * r[1], s * r[2] * r[1], s * r[1] * r[2], s * r[2] * r[2])
end

# ---------------------------------------------------------------------------------------
# Log product-integration weights
# ---------------------------------------------------------------------------------------

"""
Moments M_k(t) = ∫_{-1}^{1} log|t−s| P_k(s) ds, k = 0..n-1, for t ∈ [−1,1].
M₀ = (1+t)log(1+t) + (1−t)log(1−t) − 2; for k ≥ 1, M_k = 2(Q_{k+1}(t) − Q_{k−1}(t))/(2k+1)
with the Legendre functions of the second kind on the cut (forward recurrence, stable there).
"""
function _lq_log_moments(t::Float64, n::Int)
    M = zeros(n)
    if abs(t) == 1          # at an end: ∫ log(1 ∓ s) P_k ds = 2 log 2 − 2 (k = 0), (±1)^k (−2/(k(k + 1))) (k ≥ 1)
        M[1] = 2log(2) - 2
        for k in 1:n-1
            M[k+1] = t^k * (-2 / (k * (k + 1)))
        end
        return M
    end
    M[1] = (1 + t) * log1p(t) + (1 - t) * log1p(-t) - 2
    n == 1 && return M
    Q = zeros(n + 1)                  # Q[k+1] = Q_k(t)
    Q[1] = 0.5 * log((1 + t) / (1 - t))
    Q[2] = t * Q[1] - 1
    for k in 1:n-1
        Q[k+2] = ((2k + 1) * t * Q[k+1] - k * Q[k]) / (k + 1)
    end
    for k in 1:n-1
        M[k+1] = 2 * (Q[k+2] - Q[k]) / (2k + 1)
    end
    M
end

"""
Weights ω (length q) for the q-point Gauss–Legendre grid `qr` such that
∫_{-1}^{1} log|t−s| g(s) ds ≈ Σ ωₘ g(sₘ), exact for polynomials of degree < q.
"""
function log_product_weights(qr::Rule1D, t::Float64)
    q = length(qr)
    M = _lq_log_moments(t, q)
    ω = zeros(q)
    for m in 1:q
        acc = 0.0
        for k in 0:q-1
            acc += M[k+1] * (2k + 1) / 2 * legendre_P(k, qr.x[m])
        end
        ω[m] = acc * qr.w[m]
    end
    ω
end

# ---------------------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------------------

"""
    LineQuad(bd; tol=DEFAULT_TOL, q=nothing, qself=nothing, near_factor=nothing, α=1.0, maxdepth=60)

Precomputed data for local quadrature on the boundary `bd`:
  * `q`        Gauss–Legendre order on adaptive leaves (default ⌈p/2⌉ + q₀(tol), q₀ ≈ 11 for 1e-10)
  * `qself`    upsampled order on the self panel (default max(2p, 20))
  * `near_factor` β: panel k is near x when |x − cₖ| < β Lₖ (default from p and tol)
  * `α`        adaptive acceptance |x − γ(m)| ≥ α |γ'(m)| (b−a)
"""
struct LineQuad
    bd::BoundaryDisc
    tol::Float64
    q::Int
    qr::Rule1D
    qs::Int
    qsr::Rule1D
    selfω::Matrix{Float64}      # qs × p, column i: log product weights for singularity at sᵢ
    selfL::Matrix{Float64}      # qs × p, p-node Lagrange basis at the qs nodes
    center::Vector{Vec2}
    len::Vector{Float64}
    rnear::Vector{Float64}
    cell::Float64
    grid::Dict{Tuple{Int,Int},Vector{Int}}
    α::Float64
    maxdepth::Int
end

function LineQuad(bd::BoundaryDisc; tol::Real=DEFAULT_TOL, q=nothing, qself=nothing,
                  near_factor=nothing, α::Real=1.0, maxdepth::Int=60)
    p = bd.p
    q0 = ceil(Int, log(100 / tol) / (2 * log(3.7)))
    q === nothing && (q = cld(p, 2) + q0)
    qself === nothing && (qself = max(2p, 20))
    if near_factor === nothing
        ρs = (10 / tol)^(1 / (2p))
        a = (ρs + 1 / ρs) / 2
        near_factor = max(0.75, 1.15 * a / 2)
    end
    qr = gl_rule(q); qsr = gl_rule(qself)
    selfω = zeros(qself, p)
    for i in 1:p
        selfω[:, i] = log_product_weights(qsr, bd.rule.x[i])
    end
    selfL = interp_matrix(bd.rule, qsr.x)
    K = npanels(bd)
    center = Vector{Vec2}(undef, K); len = zeros(K)
    for k in 1:K
        P = bd.panels[k]
        center[k] = curve_eval(P, 0.0)[1]
        len[k] = sum(bd.w[panel_nodes(bd, k)])
    end
    rnear = near_factor .* len
    cell = maximum(rnear)
    grid = Dict{Tuple{Int,Int},Vector{Int}}()
    for k in 1:K
        key = (floor(Int, center[k][1] / cell), floor(Int, center[k][2] / cell))
        push!(get!(grid, key, Int[]), k)
    end
    LineQuad(bd, Float64(tol), q, qr, qself, qsr, selfω, selfL, center, len, rnear, cell, grid,
             Float64(α), maxdepth)
end

"""Panels near target x (|x − cₖ| < rnearₖ), appended to `out` (sorted)."""
function line_near_panels!(out::Vector{Int}, lq::LineQuad, x::Vec2)
    empty!(out)
    ix = floor(Int, x[1] / lq.cell); iy = floor(Int, x[2] / lq.cell)
    for di in -1:1, dj in -1:1
        v = get(lq.grid, (ix + di, iy + dj), nothing)
        v === nothing && continue
        for k in v
            norm(x - lq.center[k]) < lq.rnear[k] && push!(out, k)
        end
    end
    sort!(out)
end

"""Near-panel lists for each target point."""
line_near_lists(lq::LineQuad, targets::AbstractVector{Vec2}) =
    [line_near_panels!(Int[], lq, x) for x in targets]

# ---------------------------------------------------------------------------------------
# Accurate panel weights
# ---------------------------------------------------------------------------------------

"""
    line_panel_weights!(W, K, x, P, lq[, ℓ, stack])

Adaptive weights W[j] (2×2) with ∫_P K(x−y) ν(y) ds_y ≈ Σⱼ W[j] νⱼ, ν the degree p−1
interpolant of the nodal values on panel P. `K(r)::Mat2`, x must not lie on P (it may be
arbitrarily close). Kernel-agnostic (log, 1/ρ, smooth).
"""
function line_panel_weights!(W::AbstractVector{Mat2}, K, x::Vec2, P::CurvePanel, lq::LineQuad,
                             ℓ::Vector{Float64}=zeros(lq.bd.p),
                             stack::Vector{Tuple{Float64,Float64,Int}}=Tuple{Float64,Float64,Int}[])
    fill!(W, zero(Mat2))
    rule = lq.bd.rule; qr = lq.qr
    empty!(stack); push!(stack, (-1.0, 1.0, 0))
    while !isempty(stack)
        a, b, dep = pop!(stack)
        m = 0.5 * (a + b); h = b - a
        g, dg = curve_eval(P, m)
        if dep < lq.maxdepth && norm(x - g) < lq.α * norm(dg) * h
            push!(stack, (a, m, dep + 1)); push!(stack, (m, b, dep + 1))
            continue
        end
        for iq in eachindex(qr.x)
            s = m + 0.5 * h * qr.x[iq]
            y, dy = curve_eval(P, s)
            Kw = K(x - y) * (qr.w[iq] * 0.5 * h * norm(dy))
            lagrange_row!(ℓ, rule, s)
            @inbounds for j in eachindex(W)
                W[j] += Kw * ℓ[j]
            end
        end
    end
    W
end

"""
Self-panel weights for G2: target = node i of panel k. W[j] (2×2) with
∫_{panel k} G2(xᵢ−y) ν(y) ds ≈ Σⱼ W[j] νⱼ.
"""
function slp2d_self_panel_weights!(W::AbstractVector{Mat2}, lq::LineQuad, k::Int, i::Int; μ_2D::Real=1.0)
    bd = lq.bd; P = bd.panels[k]; qsr = lq.qsr
    fill!(W, zero(Mat2))
    si = bd.rule.x[i]
    x, dxi = curve_eval(P, si)
    c = 1 / (4π * μ_2D)
    @inbounds for m in eachindex(qsr.x)
        sm = qsr.x[m]
        y, dy = curve_eval(P, sm)
        sp = norm(dy)
        r = x - y
        ρ2 = r[1]^2 + r[2]^2
        ds = abs(si - sm)
        lsm = ds > 1e-13 ? 0.5 * log(ρ2) - log(ds) : log(norm(dxi))   # log(|r|/|sᵢ−s|)
        A = c * sp * (-lq.selfω[m, i] - qsr.w[m] * lsm)
        B = c * sp * qsr.w[m] / ρ2
        Km = Mat2(A + B * r[1]^2, B * r[1] * r[2], B * r[1] * r[2], A + B * r[2]^2)
        for j in eachindex(W)
            W[j] += Km * lq.selfL[m, j]
        end
    end
    W
end

# ---------------------------------------------------------------------------------------
# Targets on Γ: detection and the log product rule for kernels A(r) log ρ + S(r)
# ---------------------------------------------------------------------------------------

"""
    _on_curve_panel(P, x, len) -> s0 or nothing

The parameter s0 ∈ [−1, 1] with γ(s0) = x if x lies on the curve panel P (to 1e-12 of its length `len`; snapped to
±1 within 1e-12 of an end), else `nothing`: Gauss–Newton for the foot point from the closest of 9 samples.
"""
function _on_curve_panel(P::CurvePanel, x::Vec2, len::Float64)
    s = 0.0; best = Inf
    for sg in range(-1, 1; length=9)
        d = norm(curve_eval(P, sg)[1] - x)
        d < best && (best = d; s = sg)
    end
    best > 0.5len && return nothing                      # farther than half a panel from every sample
    for it in 1:30                                      # quadratic on the curve (zero residual), linear off it
        g, dg = curve_eval(P, s)
        it > 3 && norm(x - g) > 1e-3len && return nothing
        Δ = dot(x - g, dg) / dot(dg, dg)
        s = clamp(s + Δ, -1.5, 1.5)
        abs(Δ) < 1e-15 && break
    end
    abs(s) <= 1 + 1e-12 || return nothing
    s = abs(s) >= 1 - 1e-12 ? sign(s) : s
    norm(curve_eval(P, s)[1] - x) <= 1e-12 * len ? s : nothing
end

"""
The split K(r) = A(r) log ρ + S(r) of a kernel with a logarithmic singularity, A and S smooth along Γ (`τ`: the
tangent, for S's limit at r = 0): `(A, S)` as `Mat2`s, or `nothing` if the kernel has no known split.
"""
_log_split(K, r, τ) = nothing
_log_split(::LaplaceKernel2D, r, τ) = (Mat2(1 / 2π, 0, 0, 1 / 2π), zero(Mat2))
function _log_split(K::MHKernel, r, τ)                  # K0(kρ) = −I0(kρ) log ρ + (K0(kρ) + I0(kρ) log ρ)
    ρ = norm(r); z = K.k * ρ
    i0 = SpecialFunctions.besseli(0, z)
    sm = ρ == 0 ? -log(K.k / 2) - Base.MathConstants.eulergamma : SpecialFunctions.besselk(0, z) + i0 * log(ρ)
    (Mat2(-i0 / 2π, 0, 0, 0), Mat2(sm / 2π, 0, 0, 0))
end
function _log_split(K::StokesKernel, r, τ)              # G2 alone: −c log ρ I + c r rᵀ/ρ²
    isinf(K.μ_3D) || return nothing
    c = 1 / (4π * K.μ_2D); ρ2 = r[1]^2 + r[2]^2
    e = ρ2 == 0 ? τ : r / sqrt(ρ2)
    (Mat2(-c, 0, 0, -c), c * (e * e'))
end

"""Weights W[j] of the curve panel P for the target x = γ(s0) on it: the log product rule (`log_product_weights`) on
the upsampled panel rule for A log|s − s0|, plain Gauss–Legendre for A log(ρ/|s − s0|) + S."""
function _line_onpanel_weights!(W::AbstractVector{Mat2}, K, x::Vec2, s0::Float64, P::CurvePanel, lq::LineQuad)
    qsr = lq.qsr
    fill!(W, zero(Mat2))
    ω = log_product_weights(qsr, s0)
    L = lq.selfL                                        # the p-node basis at the upsampled nodes
    @inbounds for m in eachindex(qsr.x)
        sm = qsr.x[m]
        y, dy = curve_eval(P, sm)
        sp = norm(dy); r = x - y; ds = abs(sm - s0)
        A, S = _log_split(K, r, dy / sp)
        lr = ds > 1e-13 ? log(norm(r) / ds) : log(sp)
        Km = (A * (ω[m] + qsr.w[m] * lr) + S * qsr.w[m]) * sp
        for j in eachindex(W)
            W[j] += Km * L[m, j]
        end
    end
    W
end

# ---------------------------------------------------------------------------------------
# Correction assembly
# ---------------------------------------------------------------------------------------

"""A target-dependent kernel K(i, r) at target i, as a kernel of r alone (the adaptive weights take K(r))."""
struct _AtTarget{K}
    K::K
    i::Int
end
@inline (W::_AtTarget)(r::Vec2) = W.K(W.i, r)

@inline function _lq_push_block!(I, J, V, it::Int, jn::Int, C::Mat2)
    for b in 1:2, a in 1:2
        push!(I, 2(it - 1) + a); push!(J, 2(jn - 1) + b); push!(V, C[a, b])
    end
end

# Threaded loop over targets; `body!(I, J, V, W, ℓ, stack, it)` adds blocks for target it.
function _lq_threaded_assemble(body!, lq::LineQuad, nt::Int)
    p = lq.bd.p; N = length(lq.bd)
    nchunk = max(1, min(nt, 8 * Threads.nthreads()))
    ranges = collect(Iterators.partition(1:nt, cld(max(nt, 1), nchunk)))
    tasks = map(ranges) do rng
        Threads.@spawn begin
            local I = Int[]; local J = Int[]; local V = Float64[]
            W = Vector{Mat2}(undef, p); ℓ = zeros(p)
            stack = Tuple{Float64,Float64,Int}[]
            for it in rng
                body!(I, J, V, W, ℓ, stack, it)
            end
            (I, J, V)
        end
    end
    res = fetch.(tasks)
    Iall = reduce(vcat, (r[1] for r in res); init=Int[])
    Jall = reduce(vcat, (r[2] for r in res); init=Int[])
    Vall = reduce(vcat, (r[3] for r in res); init=Float64[])
    sparse(Iall, Jall, Vall, 2nt, 2N)
end

"""
    line_correction(K, lq, targets, near=line_near_lists(lq, targets))

Sparse correction (2Nₜ × 2N_Γ) for a kernel K(r)::Mat2, or K(i, r) depending on the target i (a traction kernel with
the normal at each target): accurate weights minus smooth weights K(x−yⱼ) wⱼ, on each near panel (a source node at the target itself has smooth weight 0, as in the fast sums). Off the curve the accurate
weights are adaptive. Targets on Γ (at nodes or between them, to 1e-12 of a panel length) take the log product rule
on the panels they lie on, for the kernels with a known split A log ρ + S (`LaplaceKernel2D`, `MHKernel`,
`StokesKernel` without its 3D part); another kernel that is finite at r = 0 (bounded, e.g. ρ log ρ) keeps the
adaptive weights there; an unbounded one is an `ArgumentError`.
"""
function line_correction(K, lq::LineQuad, targets::AbstractVector{Vec2},
                         near::AbstractVector=_line_near_lists_threaded(lq, targets))
    # Built in compressed form by target (columns 2it−1, 2it of the transpose; the near lists fix the
    # layout), filled by a threaded loop, then transposed: the COO triplets and `sparse` of
    # `_lq_threaded_assemble` allocated 3.5 GB and took most of the 0.9 s on the dumbbell (83k targets).
    bd = lq.bd; p = bd.p; nt = length(targets); N = length(bd)
    tdep = hasmethod(K, Tuple{Int,Vec2})
    # targets on Γ: (target, panel) → s0
    onc = Vector{Vector{Tuple{Int,Float64}}}(undef, nt)
    Threads.@threads :dynamic for it in 1:nt
        onc[it] = Tuple{Int,Float64}[]
        for k in near[it]
            s0 = _on_curve_panel(bd.panels[k], targets[it], lq.len[k])
            s0 === nothing || push!(onc[it], (k, s0))
        end
    end
    oncurve = findall(!isempty, onc)
    if !isempty(oncurve) && _log_split(K, Vec2(1, 0), Vec2(0, 1)) === nothing
        # no split: a kernel bounded at r = 0 keeps the adaptive weights on Γ; an unbounded one has no quadrature here
        bounded = all(isfinite, (tdep ? _AtTarget(K, oncurve[1]) : K)(zero(Vec2)))
        bounded || throw(ArgumentError("line_correction: targets $(oncurve[1:min(end, 10)])" *
                                       "$(length(oncurve) > 10 ? "…" : "") lie on Γ, where only kernels with a known " *
                                       "log split (LaplaceKernel2D, MHKernel, the 2D Stokeslet) or bounded at r = 0 " *
                                       "have a quadrature; $(typeof(K)) is neither"))
        foreach(empty!, onc)                      # adaptive weights everywhere, as off the curve
    end
    colptr = Vector{Int}(undef, 2nt + 1); colptr[1] = 1
    for it in 1:nt
        len = 2p * length(near[it])
        colptr[2it] = colptr[2it-1] + len; colptr[2it+1] = colptr[2it] + len
    end
    nz = colptr[end] - 1
    rowval = Vector{Int}(undef, nz); nzval = Vector{Float64}(undef, nz)
    nchunk = max(1, min(nt, 8 * Threads.nthreads()))
    @sync for rng in Iterators.partition(1:nt, cld(max(nt, 1), nchunk))
        Threads.@spawn begin
            W = Vector{Mat2}(undef, p); ℓ = zeros(p); stack = Tuple{Float64,Float64,Int}[]
            for it in rng
                x = targets[it]; p1 = colptr[2it-1]; p2 = colptr[2it]
                Kt = tdep ? _AtTarget(K, it) : K
                for (m, k) in enumerate(near[it])     # sorted panels: sorted rows
                    io = findfirst(o -> o[1] == k, onc[it])
                    if io === nothing
                        line_panel_weights!(W, Kt, x, bd.panels[k], lq, ℓ, stack)
                    else
                        _line_onpanel_weights!(W, K, x, onc[it][io][2], bd.panels[k], lq)
                    end
                    for (j, jn) in enumerate(panel_nodes(bd, k))
                        r = x - bd.x[jn]
                        C = r[1] == 0 && r[2] == 0 ? W[j] : W[j] - Kt(r) * bd.w[jn]
                        o = 2((m - 1) * p + j - 1)
                        rowval[p1+o] = 2jn - 1; nzval[p1+o] = C[1, 1]; rowval[p1+o+1] = 2jn; nzval[p1+o+1] = C[1, 2]
                        rowval[p2+o] = 2jn - 1; nzval[p2+o] = C[2, 1]; rowval[p2+o+1] = 2jn; nzval[p2+o+1] = C[2, 2]
                    end
                end
            end
        end
    end
    copy(transpose(SparseMatrixCSC(2N, 2nt, colptr, rowval, nzval)))
end

# `line_near_lists` threaded over targets
function _line_near_lists_threaded(lq::LineQuad, targets::AbstractVector{Vec2})
    near = Vector{Vector{Int}}(undef, length(targets))
    Threads.@threads :dynamic for i in eachindex(targets)
        near[i] = line_near_panels!(Int[], lq, targets[i])
    end
    near
end

"""
    slp2d_correction(lq, targets, near=line_near_lists(lq, targets); μ_2D=1.0)

Correction for S²ᴰ_Γ at off-curve targets (e.g. area nodes): exact ≈ smooth + C ν.
"""
slp2d_correction(lq::LineQuad, targets::AbstractVector{Vec2},
                 near::AbstractVector=_line_near_lists_threaded(lq, targets); μ_2D::Real=1.0) =
    line_correction(r -> _lq_G2(r, μ_2D), lq, targets, near)

"""
    slp2d_self_correction(lq; μ_2D=1.0)

Correction (2N_Γ × 2N_Γ) for S²ᴰ_Γ with the Γ nodes as targets. The smooth rule excludes
j = i; the self panel uses log product integration, other near panels adaptive quadrature.
"""
function slp2d_self_correction(lq::LineQuad; μ_2D::Real=1.0)
    bd = lq.bd; p = bd.p
    K = r -> _lq_G2(r, μ_2D)
    _lq_threaded_assemble(lq, length(bd)) do I, J, V, W, ℓ, stack, it
        x = bd.x[it]
        kself = (it - 1) ÷ p + 1; iself = it - (kself - 1) * p
        near = line_near_panels!(Int[], lq, x)
        kself in near || push!(near, kself)
        for k in near
            if k == kself
                slp2d_self_panel_weights!(W, lq, k, iself; μ_2D=μ_2D)
            else
                line_panel_weights!(W, K, x, bd.panels[k], lq, ℓ, stack)
            end
            for (j, jn) in enumerate(panel_nodes(bd, k))
                Ws = jn == it ? zero(Mat2) : K(x - bd.x[jn]) * bd.w[jn]
                _lq_push_block!(I, J, V, it, jn, W[j] - Ws)
            end
        end
    end
end

# ---------------------------------------------------------------------------------------
# Smooth rule and dense boundary operators
# ---------------------------------------------------------------------------------------

"""
Dense smooth-rule matrix (2Nₜ × 2N_Γ) Σⱼ K(x−yⱼ) wⱼ; with `self=true` (targets = Γ nodes)
the j = i entries are zero.
"""
function line_smooth_matrix(K, bd::BoundaryDisc, targets::AbstractVector{Vec2}; self::Bool=false)
    N = length(bd); nt = length(targets)
    A = zeros(2nt, 2N)
    Threads.@threads for i in 1:nt
        for j in 1:N
            (self && i == j) && continue
            B = K(targets[i] - bd.x[j]) * bd.w[j]
            A[2i-1, 2j-1] = B[1, 1]; A[2i, 2j-1] = B[2, 1]
            A[2i-1, 2j] = B[1, 2];   A[2i, 2j] = B[2, 2]
        end
    end
    A
end

"""
    slp2d_velocity_matrix(lq; μ_2D=1.0)   (or slp2d_velocity_matrix(bd; μ_2D, tol))

Dense accurate matrix (2N_Γ × 2N_Γ) of S²ᴰ_Γ restricted to Γ (interleaved layout).
"""
function slp2d_velocity_matrix(lq::LineQuad; μ_2D::Real=1.0)
    A = line_smooth_matrix(r -> _lq_G2(r, μ_2D), lq.bd, lq.bd.x; self=true)
    C = slp2d_self_correction(lq; μ_2D=μ_2D)
    rows = rowvals(C); vals = nonzeros(C)
    for j in 1:size(C, 2), idx in nzrange(C, j)
        A[rows[idx], j] += vals[idx]
    end
    A
end
slp2d_velocity_matrix(bd::BoundaryDisc; μ_2D::Real=1.0, tol::Real=DEFAULT_TOL) =
    slp2d_velocity_matrix(LineQuad(bd; tol=tol); μ_2D=μ_2D)

"""
    traction_self_matrix(bd; jump=0.5)

Dense (2N_Γ × 2N_Γ) interior-limit traction of S²ᴰ_Γ on Γ with the outward normal:
T_ΓΓ ν(xᵢ) = ½ νᵢ + Σ_{j≠i} T2(xᵢ−xⱼ, nᵢ) wⱼ νⱼ − (κᵢ/(2π)) τᵢτᵢᵀ wᵢ νᵢ.
(Independent of μ_2D. `jump = −0.5` gives the exterior limit.)
"""
function traction_self_matrix(bd::BoundaryDisc; jump::Real=0.5)
    N = length(bd)
    A = zeros(2N, 2N)
    Threads.@threads for i in 1:N
        ni = bd.n[i]
        for j in 1:N
            if j == i
                τ = bd.τ[i]
                B = Mat2(jump, 0.0, 0.0, jump) - (bd.κ[i] / (2π)) * bd.w[i] * (τ * τ')
            else
                B = _lq_T2(bd.x[i] - bd.x[j], ni) * bd.w[j]
            end
            A[2i-1, 2j-1] = B[1, 1]; A[2i, 2j-1] = B[2, 1]
            A[2i-1, 2j] = B[1, 2];   A[2i, 2j] = B[2, 2]
        end
    end
    A
end

# ---------------------------------------------------------------------------------------
# Direct accurate evaluation (small problems, tests)
# ---------------------------------------------------------------------------------------

"""
    line_potential_eval(K, lq, x, ν)

Accurate ∮_Γ K(x−y) ν(y) ds_y at an off-curve point x (ν::Vector{Vec2} nodal values): adaptive
quadrature on near panels, the smooth rule on the others.
"""
function line_potential_eval(K, lq::LineQuad, x::Vec2, ν::AbstractVector{Vec2})
    bd = lq.bd; p = bd.p
    W = Vector{Mat2}(undef, p); ℓ = zeros(p); stack = Tuple{Float64,Float64,Int}[]
    u = zero(Vec2)
    for k in 1:npanels(bd)
        idx = panel_nodes(bd, k)
        if norm(x - lq.center[k]) < lq.rnear[k]
            line_panel_weights!(W, K, x, bd.panels[k], lq, ℓ, stack)
            for (j, jn) in enumerate(idx)
                u += W[j] * ν[jn]
            end
        else
            for jn in idx
                u += K(x - bd.x[jn]) * (bd.w[jn] * ν[jn])
            end
        end
    end
    u
end

"""Accurate S²ᴰ_Γ[ν] at off-curve points."""
slp2d_eval(lq::LineQuad, targets::AbstractVector{Vec2}, ν::AbstractVector{Vec2}; μ_2D::Real=1.0) =
    [line_potential_eval(r -> _lq_G2(r, μ_2D), lq, x, ν) for x in targets]

"""Traction (normal `nx`) of S²ᴰ_Γ[ν] at an off-curve point x (independent of μ_2D)."""
slp2d_traction_eval(lq::LineQuad, x::Vec2, nx::Vec2, ν::AbstractVector{Vec2}) =
    line_potential_eval(r -> _lq_T2(r, nx), lq, x, ν)

# Interleaved flat vector <-> Vector{Vec2}
_lq_flat(v::AbstractVector{Vec2}) = collect(reinterpret(Float64, v))
_lq_vecs(v::AbstractVector{Float64}) = collect(reinterpret(Vec2, v))
