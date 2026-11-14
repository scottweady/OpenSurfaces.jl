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
    μ_2D::Float64                          # Inf: no 2D Stokeslet
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

`whole = true`: `remainder` is a tensor `SymbolKernel` K with base `Stokeslet3D(; μ_3D)` (its remainder in absolute
units, any decay), and the plan sums K itself, coincident pairs skipped (R split by weight, `_split_w`).

`μ_2D` (finite): the plan sums G2(·; μ_2D) + G3(·; μ_3D) (`StokesKernel`), the 2D Stokeslet split by Hasimoto's
screening as in `FilmEwald`; `μ_3D = Inf` then leaves G2 alone.
"""
function EwaldStokes3D(src::AbstractVector{Vec2}, trg=nothing; μ_3D::Real=1.0, remainder=nothing, tol::Real=DEFAULT_TOL,
                       σ=nothing, nneigh::Real=150, whole::Bool=false, μ_2D::Real=Inf,
                       nthreads::Int=Threads.nthreads())
    tm = Dict{Symbol,Float64}()
    has2d = isfinite(μ_2D)
    has2d || isfinite(μ_3D) || throw(ArgumentError("EwaldStokes3D: μ_3D = Inf needs a finite μ_2D"))
    isfinite(μ_3D) || remainder === nothing || throw(ArgumentError("EwaldStokes3D: a remainder needs a finite μ_3D"))
    has2d && whole && throw(ArgumentError("EwaldStokes3D: whole = true with a 2D Stokeslet is not implemented"))
    if whole
        remainder isa SymbolKernel && remainder.tensor && remainder.base isa Stokeslet3D ||
            throw(ArgumentError("EwaldStokes3D: whole = true takes a tensor SymbolKernel with base Stokeslet3D"))
        remainder.base.μ == μ_3D || throw(ArgumentError("EwaldStokes3D: μ_3D differs from the SymbolKernel's base"))
    else
        _smooth_remainder(remainder, "EwaldStokes3D")
    end
    selfmode = trg === nothing
    T = selfmode ? src : collect(Vec2, trg)
    allp = selfmode ? src : vcat(src, T)
    lo = reduce((a, b) -> min.(a, b), allp); hi = reduce((a, b) -> max.(a, b), allp)
    c0 = (lo + hi) / 2
    L = max.(hi - lo, 1e-12)
    # (the 2D short range decays like e^{−ρ²/σ²}, as FilmEwald)
    c = has2d ? max(erfcinv(tol), sqrt(log(1 / tol)), 3.0) : max(erfcinv(tol), 3.0)
    # (a new binding: reassigning the keyword would box it in the kernel closures below)
    σ0 = σ === nothing ? _calibrate_sigma(src, T, c, nneigh, sqrt(nneigh / (π * length(src) / (L[1] * L[2]))) / c) :
                         Float64(σ)
    # remainder: the grid's Nyquist wavenumber π/h = 2c/σ must be ≥ _NYQ_H/scale
    RH = remainder
    σs = RH === nothing ? σ0 : min(σ0, 2c * RH.scale / _NYQ_H)
    rc = c * σs
    whole && _check_split_range(RH, L, rc, σs, "EwaldStokes3D")
    wmin = whole ? _split_wmin(tol) : -1.0
    # window w(t; ℓ) = ½ erfc((|t| − ℓ − cσs)/σs): 1 − O(tol) for |t| ≤ ℓ, O(tol) beyond ℓ + 2cσs
    S = L .+ 2rc
    P = Tuple(2 .* S)
    h = π * σs / (2(has2d ? c + 1 : c)) / (whole ? _split_h(RH) : 1.0)   # (+1: the 2D multiplier's polynomial factor)
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
                if has2d                           # G2's long range, in the units of 4πμ_3D G3 (or alone)
                    s11, s12, s22 = _S2L(r, σs)
                    a3 = isfinite(μ_3D) ? 1.0 : 0.0; a2 = isfinite(μ_3D) ? μ_3D / μ_2D : 1.0
                    k11 = a3 * k11 + a2 * s11; k12 = a3 * k12 + a2 * s12; k22 = a3 * k22 + a2 * s22
                end
                W = wx * wy
                if RH !== nothing && W >= wmin     # 4π R_H at μ = 1 (whole: 4πμ_3D w R) (scaled by 1/(4πμ_3D) below)
                    r11, r12, r22 = remainder_tensor(RH, r)
                    s = whole ? 4π * μ_3D * _split_w(ρ, σs) : 4π
                    k11 += s * r11; k12 += s * r12; k22 += s * r22
                end
                K[a+1, b+1, 1] = W * k11; K[a+1, b+1, 2] = W * k12; K[a+1, b+1, 3] = W * k22
            end
        end
        FFTW.fft!(K, (1, 2))
        # u = (1/(P₁P₂)) Σ_m K̂(k_m) F_m e^{i k_m x} / (4πμ_3D),  K̂ ≈ h₁h₂ · FFT(samples)
        Khat = real.(K) .* (hx * hy / (P[1] * P[2] * 4π * (isfinite(μ_3D) ? μ_3D : μ_2D)))
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
    c8 = isfinite(μ_3D) ? 1 / (4π * μ_3D) : 0.0
    c2 = has2d ? 1 / (4π * μ_2D) : 0.0
    tm[:shortrange] = @elapsed Ssr = has2d ?
        _sr_matrix(src, T, selfmode, rc, (i, r) -> (ρ = norm(r); (φ, dφ) = _phiS(ρ, σs);
                                                     c8 .* _Kphi(r, φ, dφ, ρ) .+ c2 .* _S2S(r, σs))) : whole ?
        _sr_matrix(src, T, selfmode, max(rc, _split_rc(σs, tol)),
                   (i, r) -> (ρ = norm(r); (φ, dφ) = _phiS(ρ, σs); ws = _split_tail((ρ / σs)^2);
                              c8 .* _Kphi(r, φ, dφ, ρ) .+ ws .* remainder_tensor(RH, r))) :
        _sr_matrix(src, T, selfmode, rc, (i, r) -> (ρ = norm(r); (φ, dφ) = _phiS(ρ, σs); c8 .* _Kphi(r, φ, dφ, ρ)))
    W0 = (erfc((-L[1] - rc) / σs) / 2) * (erfc((-L[2] - rc) / σs) / 2)
    kl0 = has2d ? W0 * (c2 * (1 + _γE / 2 - log(σs)) + c8 * 2 / (σs * sqrt(π))) :
                  (2 / (σs * sqrt(π))) * (erfc((-L[1] - rc) / σs) / 2) * (erfc((-L[2] - rc) / σs) / 2) / (4π * μ_3D)
    selfcoef = selfmode ? kl0 : 0.0
    EwaldStokes3D(Float64(μ_3D), Float64(μ_2D), σs, rc, c0, P, M, Khat, plan1, plan2, length(src), length(T),
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

# ---------------------------------------------------------------------------------------
# A whole SymbolKernel (base + a remainder R that may decay algebraically, with a cone or a log at ρ = 0): R is split
# by the weight w(ρ) = P(m, U), U = ρ²/σ², the regularized lower incomplete gamma function: w R into the long-range
# multiplier and (1 − w) R = e^{−U} Σ_{j<m} Uʲ/j! R into the short range, out to where that is below tol. w is a sum
# of Gaussians of width σ times polynomials, so the grid of the base's split resolves it, and w ~ U^m/m! at ρ = 0
# cancels a cone of R to order 2m (w R is C^{2m}); its leftover Fourier tail is why an algebraic remainder takes a
# finer grid (`_SPLIT_REFINE`). Coincident pairs skipped: w R vanishes there, and the short range drops them (as the
# direct sum of the whole kernel, whose near corrections hold its singular part).
# ---------------------------------------------------------------------------------------

const _SPLIT_M = 4
const _SPLIT_REFINE = 2.0                      # grid refinement for an algebraic remainder (Nyquist × σ = 2c × this)
@inline function _split_w(ρ, σ)
    U = (ρ / σ)^2
    if U < _SPLIT_M                            # e^{−U} Σ_{j≥m} Uʲ/j!, without cancellation
        t = U^_SPLIT_M / factorial(_SPLIT_M); s = t
        for j in _SPLIT_M+1:_SPLIT_M+60
            t *= U / j; s += t
            t < 1e-17 * s && break
        end
        return exp(-U) * s
    end
    1 - _split_tail(U)
end
@inline function _split_tail(U)                # 1 − w = e^{−U} Σ_{j<m} Uʲ/j!
    t = 1.0; s = 1.0
    for j in 1:_SPLIT_M-1
        t *= U / j; s += t
    end
    exp(-U) * s
end
function _split_rc(σ, tol)
    U = 1.0
    while _split_tail(U) > tol / 10
        U *= 1.05
    end
    σ * sqrt(U)
end
_split_wmin(tol) = 1e-3 * tol                  # window values below this are not sampled (R is not evaluated there)
_split_h(R) = R.decay === :algebraic ? _SPLIT_REFINE : 1.0

"""The remainder must be known out to the sampled window: its table or its far field."""
function _check_split_range(R, L, rc, σ, who)
    # the window ½erfc((|t| − L − rc)/σ) falls below _split_wmin(1e-16) ≈ 1e-19 by |t| = L + rc + 6.5σ per axis
    ρneed = norm(L .+ rc .+ 6.5σ)
    R.far === nothing && ρneed > table_range(R) &&
        throw(ArgumentError("$who: the SymbolKernel's table reaches ρc = $(table_range(R)), but the long-range " *
                            "window needs its remainder out to $(round(ρneed; sigdigits=3)); build it with a larger " *
                            "`ρc` or a `far` field"))
    nothing
end

# ---------------------------------------------------------------------------------------
# The same split for the 3D Laplace kernel on coplanar points: u(xᵢ) = Σⱼ qⱼ / (4π |xᵢ − yⱼ|), with
# 1/ρ = φ_L + φ_S as above (the scalar part of the Stokeslet's K[φ]). One scalar NUFFT pair per apply, a scalar
# multiplier and a sparse short-range matrix. Coincident pairs are skipped (as FMM3D's lfmm3d and the direct sum):
# i = j in self mode and every explicit target that sits exactly on a source.
# ---------------------------------------------------------------------------------------

struct EwaldLaplace3D
    σ::Float64
    rc::Float64
    M::NTuple{2,Int}
    Khat::Matrix{Float64}                  # (M₁, M₂), FFT mode order, scaled by 1/(4π P₁P₂)
    plan1::FINUFFT.finufft_plan{Float64}   # type 1 at the sources
    plan2::FINUFFT.finufft_plan{Float64}   # type 2 at the targets
    nsrc::Int
    ntrg::Int
    selfmode::Bool
    St::SparseMatrixCSC{Float64,Int}       # short range, by target: column i holds target i's weights (no quadrature weights)
    coinc::Vector{NTuple{2,Int}}           # coincident (target, source) pairs other than i = j in self mode
    kl0::Float64                           # windowed φ_L(0)/(4π), subtracted at the coincident pairs
    timings::Dict{Symbol,Float64}
end

"""
    EwaldLaplace3D(src, trg=nothing; remainder=nothing, tol=DEFAULT_TOL, σ=nothing, nneigh=150)

Plan for u(xᵢ) = Σⱼ qⱼ / (4π|xᵢ − yⱼ|) (`LaplaceKernel3D`) with fixed source points `src` in the plane and targets
`trg` (`nothing`: targets = sources). Coincident pairs are skipped, as by FMM3D and the direct sum: i = j, explicit
targets that sit exactly on a source, and duplicated source points. σ defaults to the value giving about `nneigh`
short-range neighbors per target. Apply with `ewald_apply(E, q)` (q includes the quadrature weights).

`remainder`: a scalar `SymbolKernel` K with base `LaplaceKernel3D()`: the plan sums K itself (1/(4πρ) plus its
remainder R, which may decay algebraically), coincident pairs skipped; R is split by weight (`_split_w`).
"""
function EwaldLaplace3D(src::AbstractVector{Vec2}, trg=nothing; remainder=nothing, tol::Real=DEFAULT_TOL, σ=nothing,
                        nneigh::Real=150, nthreads::Int=Threads.nthreads())
    tm = Dict{Symbol,Float64}()
    selfmode = trg === nothing
    T = selfmode ? src : collect(Vec2, trg)
    allp = selfmode ? src : vcat(src, T)
    lo = reduce((a, b) -> min.(a, b), allp); hi = reduce((a, b) -> max.(a, b), allp)
    c0 = (lo + hi) / 2
    L = max.(hi - lo, 1e-12)
    c = max(erfcinv(tol), 3.0)
    σ0 = σ === nothing ? _calibrate_sigma(src, T, c, nneigh, sqrt(nneigh / (π * length(src) / (L[1] * L[2]))) / c) :
                         Float64(σ)
    R = remainder
    if R !== nothing
        R isa SymbolKernel && !R.tensor && R.base isa LaplaceKernel3D && :value in R.outputs ||
            throw(ArgumentError("EwaldLaplace3D: the remainder is a scalar SymbolKernel with base LaplaceKernel3D " *
                                "and the output :value"))
    end
    σs = R === nothing ? σ0 : min(σ0, 2c * R.scale / _NYQ_H)
    rc = c * σs
    S = L .+ 2rc
    P = Tuple(2 .* S)
    h = π * σs / (2c) / (R === nothing ? 1.0 : _split_h(R))
    M = ntuple(d -> 2 * ceil(Int, P[d] / (2h)), 2)
    iv = R === nothing ? 0 : findfirst(==(:value), R.outputs)
    Rv(ρ) = kernel_remainder(R, ρ)[iv]
    R === nothing || _check_split_range(R, L, rc, σs, "EwaldLaplace3D")
    tm[:multiplier] = @elapsed begin
        K = R === nothing ? _windowed_fft(r -> (_phiL(norm(r), σs)[1],), 1, M, P, L, rc, σs) :
            _windowed_fft(r -> (ρ = norm(r); (_phiL(ρ, σs)[1] + 4π * _split_w(ρ, σs) * Rv(ρ),)), 1, M, P, L, rc, σs;
                          wmin=_split_wmin(tol))
        Khat = real.(view(K, :, :, 1)) ./ 4π
    end
    tm[:plans] = @elapsed begin
        sx = [2π * (p[1] - c0[1]) / P[1] for p in src]; sy = [2π * (p[2] - c0[2]) / P[2] for p in src]
        tx = [2π * (p[1] - c0[1]) / P[1] for p in T]; ty = [2π * (p[2] - c0[2]) / P[2] for p in T]
        ntol = max(tol / 10, 100eps())     # FINUFFT's double-precision floor
        plan1 = FINUFFT.finufft_makeplan(1, [M[1], M[2]], -1, 1, ntol; modeord=1, nthreads=nthreads)
        FINUFFT.finufft_setpts!(plan1, sx, sy)
        plan2 = FINUFFT.finufft_makeplan(2, [M[1], M[2]], +1, 1, ntol; modeord=1, nthreads=nthreads)
        FINUFFT.finufft_setpts!(plan2, tx, ty)
    end
    tm[:shortrange] = @elapsed St = R === nothing ? _sr_scalar(src, T, selfmode, rc, ρ -> erfc(ρ / σs) / (4π * ρ)) :
        _sr_scalar(src, T, selfmode, max(rc, _split_rc(σs, tol)),
                   ρ -> erfc(ρ / σs) / (4π * ρ) + _split_tail((ρ / σs)^2) * Rv(ρ))
    kl0 = (2 / (σs * sqrt(π))) * (erfc((-L[1] - rc) / σs) / 2) * (erfc((-L[2] - rc) / σs) / 2) / 4π
    EwaldLaplace3D(σs, rc, M, Khat, plan1, plan2, length(src), length(T), selfmode, St,
                   _coincident_pairs(src, T, selfmode), kl0, tm)
end

"""Scalar short-range matrix by target (column i: the sources within rc of target i, coincident ones skipped)."""
function _sr_scalar(src, T, selfmode, rc, kfun)
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
            rowval[p+m] = j; nzval[p+m] = kfun(norm(x - src[j]))
        end
    end
    SparseMatrixCSC(length(src), nt, colptr, rowval, nzval)
end

"""u = Σⱼ qⱼ/(4π|xᵢ − yⱼ|) for the planned sources/targets (q already includes quadrature weights)."""
function ewald_apply(E::EwaldLaplace3D, q::AbstractVector{<:Real})
    length(q) == E.nsrc || throw(DimensionMismatch("ewald_apply: $(length(q)) strengths for $(E.nsrc) sources"))
    F = Matrix{ComplexF64}(undef, E.M[1], E.M[2])
    FINUFFT.finufft_exec!(E.plan1, complex.(Vector{Float64}(q)), F)
    F .*= E.Khat
    U = Vector{ComplexF64}(undef, E.ntrg)
    FINUFFT.finufft_exec!(E.plan2, F, U)
    St = E.St; cp = St.colptr; rv = St.rowval; nv = St.nzval
    u = Vector{Float64}(undef, E.ntrg)
    Threads.@threads for i in 1:E.ntrg
        s = real(U[i])
        @inbounds for p in cp[i]:cp[i+1]-1
            s += nv[p] * q[rv[p]]
        end
        @inbounds u[i] = E.selfmode ? s - E.kl0 * q[i] : s
    end
    @inbounds for (i, j) in E.coinc
        u[i] -= E.kl0 * q[j]
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
