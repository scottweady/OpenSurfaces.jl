# Fast in-plane 3D Stokeslet sums by Ewald splitting + NUFFT (replaces FMM3D for coplanar points).
#
#     u(xᵢ) = Σⱼ G3(xᵢ − yⱼ) fⱼ,   4πμ_3D G3(r) = K(r) = I/ρ + r rᵀ/ρ³,   r ∈ R²
# (the half-space Stokeslet at μ_3D, i.e. the full-space one at μ_3D/2).
#
# With φ = 1/ρ = φ_L + φ_S, φ_L = erf(ρ/σ)/ρ (smooth), φ_S = erfc(ρ/σ)/ρ (short range), and
# K[φ](r) := φ I − (φ′(ρ)/ρ) r rᵀ (linear in φ; K[1/ρ] = K), we have K = K[φ_L] + K[φ_S].
#
#  * Long range: K[φ_L] is smooth. Multiplied by a box window W(r) = w(r₁)w(r₂) that equals 1
#    (to tolerance) for all separations in the point set's bounding box, it has compact
#    support, so its free-space convolution equals a periodic one on a period of twice the
#    support. Its Fourier multiplier is computed once by FFT of samples; each apply is a type-1
#    NUFFT (sources → modes), a 2×2 multiplier, and a type-2 NUFFT (modes → targets).
#  * Short range: K[φ_S] decays like erfc(ρ/σ); pairs closer than r_c = c σ (erfc(c) ≈ tol)
#    are summed directly through a precomputed sparse matrix.
#  * Coincident pairs are skipped (FMM convention): the short range drops them, and the long range's
#    K[φ_L](0) f = (2/(σ√π)) f is subtracted at them, for i = j with targets = sources (`selfcoef`) and for
#    every other target that sits exactly on a source (explicit targets, duplicated points; `coinc`).

# `__init__` (src/OpenSurfaces.jl) removes FFTW.jl's threads callback, which FINUFFT's threaded FFTs
# cannot run under.

@inline function _phiL(ρ, σ)            # erf(ρ/σ)/ρ and its derivative
    x = ρ / σ
    if x < 1e-3
        c = 2 / (σ * sqrt(π))
        return c * (1 - x^2 / 3 + x^4 / 10), c * (-2x / 3 + 2x^3 / 5) / σ
    end
    v = erf(x) / ρ
    d = (2 / (σ * sqrt(π))) * exp(-x^2) / ρ - v / ρ
    v, d
end

@inline function _phiS(ρ, σ)            # erfc(ρ/σ)/ρ and its derivative
    x = ρ / σ
    v = erfc(x) / ρ
    d = -(2 / (σ * sqrt(π))) * exp(-x^2) / ρ - v / ρ
    v, d
end

"""K[φ](r) = φ I − (φ′/ρ) r rᵀ, returned as (K11, K12, K22)."""
@inline function _Kphi(r::Vec2, φ, dφ, ρ)
    a = ρ > 0 ? dφ / ρ : 0.0
    (φ - a * r[1]^2, -a * r[1] * r[2], φ - a * r[2]^2)
end

struct EwaldStokes3D
    μ_3D::Float64
    σ::Float64
    rc::Float64
    c0::Vec2
    P::NTuple{2,Float64}
    M::NTuple{2,Int}
    Khat::Array{Float64,3}                 # (M₁, M₂, 3): K11, K12, K22, FFT mode order, scaled
    plan1::FINUFFT.finufft_plan{Float64}   # type 1 at the sources
    plan2::FINUFFT.finufft_plan{Float64}   # type 2 at the targets
    nsrc::Int
    ntrg::Int
    selfmode::Bool
    Ssr::SRMat                             # short-range part (no weights)
    selfcoef::Float64                      # windowed K[φ_L](0)/(4πμ_3D), subtracted at i = j (targets = sources)
    coinc::Vector{NTuple{2,Int}}           # other coincident (target, source) pairs, where it is subtracted too
    kl0::Float64                           # windowed K[φ_L](0)/(4πμ_3D)
    timings::Dict{Symbol,Float64}
end

"""
    EwaldStokes3D(src, trg=nothing; μ_3D=1, remainder=nothing, tol=DEFAULT_TOL, σ=nothing, nneigh=150)

Plan for u(xᵢ) = Σⱼ G3(xᵢ − yⱼ) fⱼ (`G3` the half-space Stokeslet at `μ_3D`; for an unbounded fluid of viscosity μ_f, pass μ_3D = 2μ_f) with fixed
source points `src` and targets `trg`
(`nothing`: targets = sources). Coincident pairs are skipped, as by FMM3D and the direct sums: i = j, explicit targets
that sit exactly on a source, and duplicated source points. σ defaults to the value giving about
`nneigh` short-range neighbors per target. A `remainder` (a tensor `SymbolKernel` with base `Stokeslet3D()`, at
μ = 1; e.g. a finite depth, R_H = G_H − G3) is added to the long-range multiplier, all pairs including i = j, with σ
small enough for the grid to resolve it on its `scale`.
"""
function EwaldStokes3D(src::AbstractVector{Vec2}, trg=nothing; μ_3D::Real=1.0, remainder=nothing, tol::Real=DEFAULT_TOL,
                       σ=nothing, nneigh::Real=150, nthreads::Int=Threads.nthreads())
    tm = Dict{Symbol,Float64}()
    _smooth_remainder(remainder, "EwaldStokes3D")
    selfmode = trg === nothing
    T = selfmode ? src : collect(Vec2, trg)
    allp = selfmode ? src : vcat(src, T)
    lo = reduce((a, b) -> min.(a, b), allp); hi = reduce((a, b) -> max.(a, b), allp)
    c0 = (lo + hi) / 2
    L = max.(hi - lo, 1e-12)
    c = max(erfcinv(tol), 3.0)
    # (a new binding: reassigning the keyword would box it in the kernel closures below)
    σ0 = σ === nothing ? _calibrate_sigma(src, T, c, nneigh, sqrt(nneigh / (π * length(src) / (L[1] * L[2]))) / c) :
                         Float64(σ)
    # remainder: the grid's Nyquist wavenumber π/h = 2c/σ must be ≥ _NYQ_H/scale
    RH = remainder
    σs = RH === nothing ? σ0 : min(σ0, 2c * RH.scale / _NYQ_H)
    rc = c * σs
    # window w(t; ℓ) = ½ erfc((|t| − ℓ − cσs)/σs): 1 − O(tol) for |t| ≤ ℓ, O(tol) beyond ℓ + 2cσs
    S = L .+ 2rc
    P = Tuple(2 .* S)
    h = π * σs / (2c)
    M = ntuple(d -> 2 * ceil(Int, P[d] / (2h)), 2)
    hx = P[1] / M[1]; hy = P[2] / M[2]
    tm[:multiplier] = @elapsed begin
        K = zeros(ComplexF64, M[1], M[2], 3)
        Threads.@threads for b in 0:M[2]-1
            ry = (b < M[2] ÷ 2 ? b : b - M[2]) * hy
            wy = erfc((abs(ry) - L[2] - rc) / σs) / 2
            for a in 0:M[1]-1
                rx = (a < M[1] ÷ 2 ? a : a - M[1]) * hx
                wx = erfc((abs(rx) - L[1] - rc) / σs) / 2
                r = Vec2(rx, ry); ρ = norm(r)
                φ, dφ = _phiL(ρ, σs)
                k11, k12, k22 = _Kphi(r, φ, dφ, ρ)
                if RH !== nothing                  # 4π R_H at μ = 1 (scaled by 1/(4πμ_3D) below)
                    r11, r12, r22 = remainder_tensor(RH, r)
                    k11 += 4π * r11; k12 += 4π * r12; k22 += 4π * r22
                end
                W = wx * wy
                K[a+1, b+1, 1] = W * k11; K[a+1, b+1, 2] = W * k12; K[a+1, b+1, 3] = W * k22
            end
        end
        FFTW.fft!(K, (1, 2))
        # u = (1/(P₁P₂)) Σ_m K̂(k_m) F_m e^{i k_m x} / (4πμ_3D),  K̂ ≈ h₁h₂ · FFT(samples)
        Khat = real.(K) .* (hx * hy / (P[1] * P[2] * 4π * μ_3D))
    end
    tm[:plans] = @elapsed begin
        sx = [2π * (p[1] - c0[1]) / P[1] for p in src]; sy = [2π * (p[2] - c0[2]) / P[2] for p in src]
        tx = [2π * (p[1] - c0[1]) / P[1] for p in T]; ty = [2π * (p[2] - c0[2]) / P[2] for p in T]
        ntol = max(tol / 10, 100eps())     # FINUFFT's double-precision floor
        plan1 = FINUFFT.finufft_makeplan(1, [M[1], M[2]], -1, 2, ntol; modeord=1, nthreads=nthreads)
        FINUFFT.finufft_setpts!(plan1, sx, sy)
        plan2 = FINUFFT.finufft_makeplan(2, [M[1], M[2]], +1, 2, ntol; modeord=1, nthreads=nthreads)
        FINUFFT.finufft_setpts!(plan2, tx, ty)
    end
    c8 = 1 / (4π * μ_3D)
    tm[:shortrange] = @elapsed Ssr = _sr_matrix(src, T, selfmode, rc,
        (i, r) -> (ρ = norm(r); (φ, dφ) = _phiS(ρ, σs); c8 .* _Kphi(r, φ, dφ, ρ)))
    kl0 = (2 / (σs * sqrt(π))) * (erfc((-L[1] - rc) / σs) / 2) * (erfc((-L[2] - rc) / σs) / 2) / (4π * μ_3D)
    selfcoef = selfmode ? kl0 : 0.0
    EwaldStokes3D(Float64(μ_3D), σs, rc, c0, P, M, Khat, plan1, plan2, length(src), length(T),
                  selfmode, Ssr, selfcoef, _coincident_pairs(src, T, selfmode), kl0, tm)
end

"""u = Σⱼ G3(xᵢ − yⱼ) fⱼ for the planned sources/targets (f already includes quadrature weights)."""
function ewald_apply(E::EwaldStokes3D, f::AbstractVector{Vec2})
    ns = E.nsrc
    c = Matrix{ComplexF64}(undef, ns, 2)
    @inbounds for j in 1:ns
        c[j, 1] = f[j][1]; c[j, 2] = f[j][2]
    end
    F = Array{ComplexF64}(undef, E.M[1], E.M[2], 2)
    FINUFFT.finufft_exec!(E.plan1, c, F)
    K = E.Khat
    @inbounds Threads.@threads for b in 1:E.M[2]
        for a in 1:E.M[1]
            F1 = F[a, b, 1]; F2 = F[a, b, 2]
            F[a, b, 1] = K[a, b, 1] * F1 + K[a, b, 2] * F2
            F[a, b, 2] = K[a, b, 2] * F1 + K[a, b, 3] * F2
        end
    end
    U = Matrix{ComplexF64}(undef, E.ntrg, 2)
    FINUFFT.finufft_exec!(E.plan2, F, U)
    fi = interleave(f)
    usr = E.Ssr * fi
    u = Vector{Vec2}(undef, E.ntrg)
    @inbounds for i in 1:E.ntrg
        v = Vec2(real(U[i, 1]) + usr[2i-1], real(U[i, 2]) + usr[2i])
        E.selfmode && (v -= E.selfcoef * f[i])
        u[i] = v
    end
    @inbounds for (i, j) in E.coinc
        u[i] -= E.kl0 * f[j]
    end
    u
end

"""An algebraically decaying remainder has a cone or a log at ρ = 0, which a long-range multiplier would sample."""
_smooth_remainder(R, who) = R === nothing || R.decay !== :algebraic ||
    throw(ArgumentError("$who: the remainder decays algebraically in k (a cone or a log at ρ = 0), which the long-range " *
                        "multiplier cannot resolve; use the SymbolKernel as the kernel of an Evaluator (fast = :direct)"))

"""The (target, source) pairs at the same point (exact match), without i = j when the targets are the sources."""
function _coincident_pairs(src, T, selfmode::Bool)
    at = Dict{Vec2,Vector{Int}}()                      # point → the sources there
    @inbounds for j in eachindex(src)
        push!(get!(() -> Int[], at, _hkey(src[j])), j)
    end
    pairs = NTuple{2,Int}[]
    @inbounds for i in eachindex(T)
        js = get(at, _hkey(T[i]), nothing)
        js === nothing && continue
        for j in js
            (selfmode && i == j) || push!(pairs, (i, j))
        end
    end
    pairs
end

# ---------------------------------------------------------------------------------------
# Convolution with a smooth kernel (the finite-depth remainders of the inviscid film): the long-range
# half of the Ewald plan alone. The windowed kernel is sampled on the doubled periodic grid and FFT'd
# once; each apply is a type-1 NUFFT, an nout × nin multiplier and a type-2 NUFFT. All pairs, i = j
# included. The kernel varies on the length `scale`: grid Nyquist wavenumber ≥ _NYQ_H/scale, window width
# c·scale/10.
# ---------------------------------------------------------------------------------------

struct SmoothConvolution
    nin::Int
    nout::Int
    M::NTuple{2,Int}
    Khat::Array{ComplexF64,4}              # (M₁, M₂, nout, nin), FFT mode order, scaled
    plan1::FINUFFT.finufft_plan{Float64}
    plan2::FINUFFT.finufft_plan{Float64}
    nsrc::Int
    ntrg::Int
end

"""
    SmoothConvolution(kfun, nout, nin, src, trg=nothing; scale, tol=DEFAULT_TOL)

Plan for uᵢ = Σⱼ K(xᵢ − yⱼ) fⱼ, all pairs, with K(r) an nout × nin matrix that is smooth on the length `scale`;
`kfun(r)` returns its entries in column-major order. `trg = nothing`: targets = sources.
"""
function SmoothConvolution(kfun, nout::Int, nin::Int, src::AbstractVector{Vec2}, trg=nothing; scale::Real, tol::Real=DEFAULT_TOL,
                           nthreads::Int=Threads.nthreads())
    T = trg === nothing ? src : collect(Vec2, trg)
    allp = trg === nothing ? src : vcat(src, T)
    lo = reduce((a, b) -> min.(a, b), allp); hi = reduce((a, b) -> max.(a, b), allp)
    c0 = (lo + hi) / 2; L = max.(hi - lo, 1e-12)
    c = max(erfcinv(tol), 3.0)
    σw = 2c * scale / _NYQ_H; rc = c * σw
    P = Tuple(2 .* (L .+ 2rc))
    h = π * σw / (2c)
    M = ntuple(d -> 2 * ceil(Int, P[d] / (2h)), 2)
    K = _windowed_fft(r -> kfun(r), nout * nin, M, P, L, rc, σw)
    Khat = reshape(K, M[1], M[2], nout, nin)
    sx = [2π * (p[1] - c0[1]) / P[1] for p in src]; sy = [2π * (p[2] - c0[2]) / P[2] for p in src]
    tx = [2π * (p[1] - c0[1]) / P[1] for p in T]; ty = [2π * (p[2] - c0[2]) / P[2] for p in T]
    ntol = max(tol / 10, 100eps())     # FINUFFT's double-precision floor
    plan1 = FINUFFT.finufft_makeplan(1, [M[1], M[2]], -1, nin, ntol; modeord=1, nthreads=nthreads)
    FINUFFT.finufft_setpts!(plan1, sx, sy)
    plan2 = FINUFFT.finufft_makeplan(2, [M[1], M[2]], +1, nout, ntol; modeord=1, nthreads=nthreads)
    FINUFFT.finufft_setpts!(plan2, tx, ty)
    SmoothConvolution(nin, nout, M, Khat, plan1, plan2, length(src), length(T))
end

"""u = Σⱼ K(xᵢ − yⱼ) fⱼ: `f` is nsrc × nin (weights included), the result ntrg × nout."""
function smooth_apply(S::SmoothConvolution, f::AbstractMatrix{<:Real})
    F = Array{ComplexF64}(undef, S.M[1], S.M[2], S.nin)
    FINUFFT.finufft_exec!(S.plan1, Matrix{ComplexF64}(f), F)
    G = zeros(ComplexF64, S.M[1], S.M[2], S.nout)
    K = S.Khat
    @inbounds Threads.@threads for b in 1:S.M[2]
        for i in 1:S.nin, o in 1:S.nout, a in 1:S.M[1]
            G[a, b, o] += K[a, b, o, i] * F[a, b, i]
        end
    end
    U = Matrix{ComplexF64}(undef, S.ntrg, S.nout)
    FINUFFT.finufft_exec!(S.plan2, G, U)
    real.(U)
end
smooth_apply(S::SmoothConvolution, f::AbstractVector{<:Real}) = vec(smooth_apply(S, reshape(f, :, 1)))
