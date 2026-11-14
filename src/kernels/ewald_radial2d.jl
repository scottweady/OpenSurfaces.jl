# Ewald + NUFFT sums of the 2D radial kernels of the film's volume potentials, for points in the plane:
#   :log   G(ρ) = (1/2π) log ρ          (LaplaceKernel2D; LaplaceGradient2D is its gradient; BiharmonicGradient2D is
#                                        three such sums and moments, as on the FMM path)
#   :mh    G(ρ) = K₀(kρ)/2π             (MHKernel; MHNormalKernel is its gradient at the target, dotted with nₓ)
# Both are heat-kernel integrals, G(ρ) = ∓(1/4π) ∫₀^∞ e^{−s k² − ρ²/(4s)} ds/s (k = 0 for log, up to a constant), and
# the split is at s = σ²/4:
#   long range  G_L: s ≥ σ²/4, Fourier transform ∓e^{−σ²(k² + |ξ|²)/4}/(k² + |ξ|²), smooth on the length σ;
#   short range G_S = G − G_L: s < σ²/4, decays like E₁(ρ²/σ²) (below `tol` beyond r_c = c σ).
# For log ρ these are closed forms (G_S = −E₁(U)/4π, U = ρ²/σ²); for K₀ the long part is tabulated once per plan on
# ρ ≤ ρ_T = √40 σ (Chebyshev panels, from the integral in t = log(4s/σ²)), and equals G (to 1e−19) beyond.
# The long part is windowed and FFT'd on the doubled grid as in ewald.jl (`_windowed_fft`); a gradient multiplies the
# multiplier by iξ. Coincident pairs are skipped (as the FMM and the direct sums): the short range drops them, and the
# long range's value G_L(0) is subtracted at them (its gradient vanishes there).

const _RAD_UT = 40.0                       # U = ρ²/σ² beyond which G_S < E₁(40)/4π ≈ 8e-20: G_L = G
const _RAD_PW = 0.5                        # table panel width / σ
const _RAD_NC = 20                         # Chebyshev nodes per table panel

"""(1/4π)∫ over t ≥ 0 of e^{−a eᵗ − U e^{−t}} × (1, e^{−t}): the long part of K₀ and its ρ-derivative / ρ (unscaled)."""
function _mh_long_integrals(a, U)
    T = a < 60 ? log(60 / a) + 2 : 2.0
    np = ceil(Int, T / 0.25)
    x, w = gausslegendre(20)
    s0 = 0.0; s1 = 0.0
    for p in 0:np-1, j in eachindex(x)
        t = (p + (x[j] + 1) / 2) * (T / np)
        e = exp(-a * exp(t) - U * exp(-t)) * w[j] * (T / np) / 2
        s0 += e; s1 += e * exp(-t)
    end
    s0 / 4π, s1 / 4π
end

"""Chebyshev table of K₀'s long part G_L(ρ) and G_L′(ρ)/ρ on [0, ρ_T]."""
struct _MHLongTable
    σ::Float64
    k::Float64
    np::Int
    c::Array{Float64,3}                    # (_RAD_NC, np, 2)
end
function _MHLongTable(k::Float64, σ::Float64)
    nc = _RAD_NC; ρT = sqrt(_RAD_UT) * σ; h = _RAD_PW * σ
    np = ceil(Int, ρT / h)
    t = [cos(π * (j - 0.5) / nc) for j in 1:nc]
    Ti = inv([cos((m - 1) * π * (j - 0.5) / nc) for j in 1:nc, m in 1:nc])
    a = (k * σ)^2 / 4
    c = zeros(nc, np, 2)
    Threads.@threads for p in 1:np
        v = zeros(nc, 2)
        for j in 1:nc
            ρ = h * (p - 1 + (t[j] + 1) / 2)
            s0, s1 = _mh_long_integrals(a, (ρ / σ)^2)
            v[j, 1] = s0; v[j, 2] = -2s1 / σ^2           # dG_L/dρ = −(2ρ/σ²) s1
        end
        c[:, p, 1] = Ti * v[:, 1]; c[:, p, 2] = Ti * v[:, 2]
    end
    _MHLongTable(σ, k, np, c)
end
"""(G_L(ρ), G_L′(ρ)/ρ) of K₀(kρ)/2π."""
@inline function (T::_MHLongTable)(ρ::Float64)
    σ = T.σ
    if ρ >= sqrt(_RAD_UT) * σ
        K0, zK1 = _mhK01(T.k * ρ)
        return K0 / 2π, -zK1 / (2π * ρ^2)
    end
    h = _RAD_PW * σ
    p = clamp(floor(Int, ρ / h) + 1, 1, T.np)
    x = 2 * (ρ / h - (p - 1)) - 1
    b1 = 0.0; b2 = 0.0; d1 = 0.0; d2 = 0.0
    @inbounds for m in _RAD_NC:-1:2
        b1, b2 = 2x * b1 - b2 + T.c[m, p, 1], b1
        d1, d2 = 2x * d1 - d2 + T.c[m, p, 2], d1
    end
    @inbounds (x * b1 - b2 + T.c[1, p, 1], x * d1 - d2 + T.c[1, p, 2])
end

# (1/2π)(log ρ + E₁(U)/2), U = ρ²/σ²: the long part of (1/2π) log ρ (series below U = 1/2)
@inline function _log_long(ρ, σ)
    U = (ρ / σ)^2
    if U < 0.5
        s = 0.0; term = 1.0
        for k in 1:30
            term *= -U / k
            s += term / k
            abs(term) < 1e-17 && break
        end
        return (log(σ) - _γE / 2 - s / 2) / 2π
    end
    U > _RAD_UT && return log(ρ) / 2π
    (log(ρ) + expint(1, U) / 2) / 2π
end

"""
    RadialEwald2D(kind, src, trg=nothing; k=0, out=:value, nd=1, tol=DEFAULT_TOL, σ=nothing, nneigh=150)

Plan for uᵢ = Σⱼ G(xᵢ − yⱼ) qⱼ with G = (1/2π) log ρ (`kind = :log`) or K₀(kρ)/2π (`:mh`, wavenumber `k`), sources
`src`, targets `trg` (`nothing`: targets = sources), coincident pairs skipped. `out = :value` (`nd` densities at once:
q is nsrc × nd) or `:gradient` (∇ₓ, one density; the result a `Vec2` per target). Apply with `ewald_apply(E, q)`.
"""
struct RadialEwald2D
    kind::Symbol
    k::Float64
    out::Symbol
    nd::Int
    σ::Float64
    rc::Float64
    M::NTuple{2,Int}
    P::NTuple{2,Float64}
    Khat::Matrix{Float64}                  # (M₁, M₂), FFT mode order, scaled
    plan1::FINUFFT.finufft_plan{Float64}
    plan2::FINUFFT.finufft_plan{Float64}
    nsrc::Int
    ntrg::Int
    selfmode::Bool
    St::SparseMatrixCSC{Float64,Int}       # short range by target: value, or ∂₁ of the gradient
    St2::Union{Nothing,SparseMatrixCSC{Float64,Int}}   # ∂₂ of the gradient
    coinc::Vector{NTuple{2,Int}}
    gl0::Float64                           # windowed G_L(0), subtracted at coincident pairs (values)
    timings::Dict{Symbol,Float64}
end

function RadialEwald2D(kind::Symbol, src::AbstractVector{Vec2}, trg=nothing; k::Real=0.0, out::Symbol=:value,
                       nd::Int=1, tol::Real=DEFAULT_TOL, σ=nothing, nneigh::Real=150, nthreads::Int=Threads.nthreads())
    kind in (:log, :mh) || throw(ArgumentError("RadialEwald2D: kind is :log or :mh"))
    out in (:value, :gradient) || throw(ArgumentError("RadialEwald2D: out is :value or :gradient"))
    out === :gradient && nd != 1 && throw(ArgumentError("RadialEwald2D: one density for a gradient"))
    kind === :mh && !(k > 0) && throw(ArgumentError("RadialEwald2D: k > 0 for :mh"))
    tm = Dict{Symbol,Float64}()
    selfmode = trg === nothing
    T = selfmode ? src : collect(Vec2, trg)
    allp = selfmode ? src : vcat(src, T)
    lo = reduce((a, b) -> min.(a, b), allp); hi = reduce((a, b) -> max.(a, b), allp)
    c0 = (lo + hi) / 2
    L = max.(hi - lo, 1e-12)
    c = max(erfcinv(tol), 3.0)
    σs = σ === nothing ? _calibrate_sigma(src, T, c, nneigh, sqrt(nneigh / (π * length(src) / (L[1] * L[2]))) / c) :
                         Float64(σ)
    rc = c * σs
    P = Tuple(2 .* (L .+ 2rc))
    h = π * σs / (2c)
    M = ntuple(d -> 2 * ceil(Int, P[d] / (2h)), 2)
    kf = Float64(k)
    tm[:table] = @elapsed (tab = kind === :mh ? _MHLongTable(kf, σs) : nothing)
    GL(ρ) = kind === :log ? _log_long(ρ, σs) : tab(ρ)[1]
    tm[:multiplier] = @elapsed begin
        K = _windowed_fft(r -> (GL(norm(r)),), 1, M, P, L, rc, σs)
        Khat = real.(view(K, :, :, 1))
    end
    tm[:plans] = @elapsed begin
        sx = [2π * (p[1] - c0[1]) / P[1] for p in src]; sy = [2π * (p[2] - c0[2]) / P[2] for p in src]
        tx = [2π * (p[1] - c0[1]) / P[1] for p in T]; ty = [2π * (p[2] - c0[2]) / P[2] for p in T]
        ntol = max(tol / 10, 100eps())
        plan1 = FINUFFT.finufft_makeplan(1, [M[1], M[2]], -1, nd, ntol; modeord=1, nthreads=nthreads)
        FINUFFT.finufft_setpts!(plan1, sx, sy)
        plan2 = FINUFFT.finufft_makeplan(2, [M[1], M[2]], +1, out === :value ? nd : 2, ntol; modeord=1, nthreads=nthreads)
        FINUFFT.finufft_setpts!(plan2, tx, ty)
    end
    tm[:shortrange] = @elapsed begin
        if out === :value
            GS = kind === :log ? (ρ -> -expint(1, (ρ / σs)^2) / 4π) : (ρ -> _mhK01(kf * ρ)[1] / 2π - tab(ρ)[1])
            St = _sr_scalar(src, T, selfmode, rc, GS); St2 = nothing
        else
            # ∇G_S = r × (G′ − G_L′)/ρ
            dS = kind === :log ? (ρ -> exp(-(ρ / σs)^2) / (2π * ρ^2)) :
                                 (ρ -> -_mhK01(kf * ρ)[2] / (2π * ρ^2) - tab(ρ)[2])
            St = _sr_scalar_r(src, T, selfmode, rc, (r, ρ) -> r[1] * dS(ρ))
            St2 = _sr_scalar_r(src, T, selfmode, rc, (r, ρ) -> r[2] * dS(ρ))
        end
    end
    W0 = (erfc((-L[1] - rc) / σs) / 2) * (erfc((-L[2] - rc) / σs) / 2)
    RadialEwald2D(kind, kf, out, nd, σs, rc, M, P, Khat, plan1, plan2, length(src), length(T), selfmode, St, St2,
                  _coincident_pairs(src, T, selfmode), W0 * GL(0.0), tm)
end

"""`_sr_scalar` with a kernel of the separation vector, `kfun(r, ρ)`."""
function _sr_scalar_r(src, T, selfmode, rc, kfun)
    cells, cell = _cell_list(src, rc)
    nt = length(T); rc2 = rc^2
    cnt = zeros(Int, nt)
    _chunked_foreach(nt) do b, i
        _neighbors!(b, T[i], src, cells, cell, rc2, selfmode ? i : 0)
        cnt[i] = length(b)
    end
    colptr = Vector{Int}(undef, nt + 1); colptr[1] = 1
    for i in 1:nt
        colptr[i+1] = colptr[i] + cnt[i]
    end
    rowval = Vector{Int}(undef, colptr[end] - 1); nzval = Vector{Float64}(undef, colptr[end] - 1)
    _chunked_foreach(nt) do b, i
        _neighbors!(b, T[i], src, cells, cell, rc2, selfmode ? i : 0)
        p = colptr[i] - 1; x = T[i]
        @inbounds for (m, j) in enumerate(b)
            r = x - src[j]
            rowval[p+m] = j; nzval[p+m] = kfun(r, norm(r))
        end
    end
    SparseMatrixCSC(length(src), nt, colptr, rowval, nzval)
end

# Σ_p St[p, i] q[rv[p], d] per target, threaded
function _sr_apply!(u::AbstractMatrix, St::SparseMatrixCSC, q::AbstractMatrix, col::Int)
    cp = St.colptr; rv = St.rowval; nv = St.nzval
    Threads.@threads for i in 1:size(St, 2)
        for d in axes(q, 2)
            s = 0.0
            @inbounds for p in cp[i]:cp[i+1]-1
                s += nv[p] * q[rv[p], d]
            end
            @inbounds u[i, col + d - 1] += s
        end
    end
    u
end

"""Values (ntrg × nd, or a vector for nd = 1) or gradients (`Vec2` per target) for strengths `q` (weights included)."""
function ewald_apply(E::RadialEwald2D, q::AbstractVecOrMat{<:Real})
    Q = reshape(Matrix{Float64}(reshape(q, E.nsrc, :)), E.nsrc, :)
    size(Q, 2) == E.nd || throw(DimensionMismatch("ewald_apply: $(size(Q, 2)) densities for a plan of $(E.nd)"))
    F = Array{ComplexF64}(undef, E.M[1], E.M[2], E.nd)
    FINUFFT.finufft_exec!(E.plan1, Matrix{ComplexF64}(Q), F)
    if E.out === :value
        F .*= E.Khat
        U = Matrix{ComplexF64}(undef, E.ntrg, E.nd)
        FINUFFT.finufft_exec!(E.plan2, F, U)
        u = real.(U)
        _sr_apply!(u, E.St, Q, 1)
        @inbounds for d in 1:E.nd
            if E.selfmode
                for i in 1:E.ntrg
                    u[i, d] -= E.gl0 * Q[i, d]
                end
            end
            for (i, j) in E.coinc
                u[i, d] -= E.gl0 * Q[j, d]
            end
        end
        return E.nd == 1 ? vec(u) : u
    end
    G = Array{ComplexF64}(undef, E.M[1], E.M[2], 2)
    M1, M2 = E.M; P1, P2 = E.P
    @inbounds Threads.@threads for b in 1:M2
        m2 = b - 1 < M2 ÷ 2 ? b - 1 : b - 1 - M2
        k2 = m2 == -M2 ÷ 2 ? 0.0 : 2π * m2 / P2
        for a in 1:M1
            m1 = a - 1 < M1 ÷ 2 ? a - 1 : a - 1 - M1
            k1 = m1 == -M1 ÷ 2 ? 0.0 : 2π * m1 / P1
            v = E.Khat[a, b] * F[a, b, 1]
            G[a, b, 1] = im * k1 * v; G[a, b, 2] = im * k2 * v
        end
    end
    U = Matrix{ComplexF64}(undef, E.ntrg, 2)
    FINUFFT.finufft_exec!(E.plan2, G, U)
    u = real.(U)
    _sr_apply!(u, E.St, Q, 1); _sr_apply!(u, E.St2, Q, 2)
    [Vec2(u[i, 1], u[i, 2]) for i in 1:E.ntrg]
end
