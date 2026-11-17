# Ewald/NUFFT evaluation of all far-field sums of the film operator on one grid:
#
#   area → area velocity:  Σⱼ [G2 + G3](xᵢ − yⱼ) fⱼ           (self term excluded)
#   Γ → area velocity:     Σⱼ G2(xᵢ − zⱼ) gⱼ
#   area → Γ traction:     Σⱼ σ2(zᵢ − yⱼ)[fⱼ] · n(zᵢ)
#
# 2D Stokeslet split (Hasimoto screening of the biharmonic, U = ρ²/σ²):
#   S = −log ρ I + rrᵀ/ρ² = S_L + S_S,  S_S = (½E₁(U) − e^{−U}) I + e^{−U} rrᵀ/ρ²,
#   G2 = S/(4π μ_2D).
# 2D Stokeslet stress σ2[f] = −(1/π) r (r·f) rᵀ/ρ⁴ = T_L + T_S,
#   T_S[f] = (e^{−U}/2π) [ (r·f) I/σ² + (r fᵀ + f rᵀ)/σ² − 2 (1/(ρ²σ²) + 1/ρ⁴) rrᵀ (r·f) ].
# 3D in-plane Stokeslet as in ewald.jl (erf/erfc split of 1/ρ).
# The smooth parts are windowed to the bounding box and turned into Fourier multipliers by FFT
# of samples (see ewald.jl); the short-range parts are sparse matrices.


const _γE = 0.5772156649015329

# −log ρ − ½E₁(U) + e^{−U} (the I-coefficient of S_L), stable for small U
@inline function _sL_a(ρ, σ)
    U = (ρ / σ)^2
    if U < 0.5
        s = 0.0; term = 1.0
        for k in 1:30
            term *= -U / k
            s += term / k
            abs(term) < 1e-17 && break
        end
        return _γE / 2 - log(σ) + s / 2 + exp(-U)
    end
    -log(ρ) - expint(1, U) / 2 + exp(-U)
end

# (1 − e^{−U})/ρ² (the rrᵀ-coefficient of S_L), stable
@inline _sL_b(ρ, σ) = (U = (ρ / σ)^2; U == 0 ? 1 / σ^2 : -expm1(-U) / (U * σ^2))

"""Long-range 2D Stokeslet S_L(r) as (S11, S12, S22) (unnormalized; G2 = S/(4π μ_2D))."""
@inline function _S2L(r::Vec2, σ)
    ρ = norm(r)
    a = _sL_a(ρ, σ); b = _sL_b(ρ, σ)
    (a + b * r[1]^2, b * r[1] * r[2], a + b * r[2]^2)
end

"""Short-range 2D Stokeslet S_S(r) as (S11, S12, S22)."""
@inline function _S2S(r::Vec2, σ)
    ρ2 = r[1]^2 + r[2]^2; U = ρ2 / σ^2; e = exp(-U)
    a = expint(1, U) / 2 - e; b = e / ρ2
    (a + b * r[1]^2, b * r[1] * r[2], a + b * r[2]^2)
end

# stress kernels: for a force f, σ[f] = Σ_l T^{(l)} f_l; returned as
# (σ11⁽¹⁾, σ12⁽¹⁾, σ22⁽¹⁾, σ11⁽²⁾, σ12⁽²⁾, σ22⁽²⁾)
@inline function _stress_terms(r::Vec2, cI, cS, cR)
    # σ[f] = cI (r·f) I + cS (r fᵀ + f rᵀ) + cR rrᵀ (r·f)
    x, y = r[1], r[2]
    (cI * x + 2cS * x + cR * x * x * x,   # f = e₁: r·f = x
     cS * y + cR * x * y * x,
     cI * x + cR * y * y * x,
     cI * y + cR * x * x * y,             # f = e₂: r·f = y
     cS * x + cR * x * y * y,
     cI * y + 2cS * y + cR * y * y * y)
end

@inline function _T2S(r::Vec2, σ)
    ρ2 = r[1]^2 + r[2]^2; U = ρ2 / σ^2; e = exp(-U) / (2π)
    _stress_terms(r, e / σ^2, e / σ^2, -2e * (1 / (ρ2 * σ^2) + 1 / ρ2^2))
end

@inline function _T2L(r::Vec2, σ)
    ρ2 = r[1]^2 + r[2]^2
    ρ2 == 0 && return ntuple(_ -> 0.0, 6)
    U = ρ2 / σ^2; e = exp(-U) / (2π)
    g = -expm1(-U) / U                       # (1 − e^{−U})/U
    # T_L = T − T_S = (1/2π)[−2 rr(r·f)(1 − e^{−U})/ρ⁴ − e^{−U}((r·f)I + rf + fr − 2rr(r·f)/ρ²)/σ²]
    _stress_terms(r, -e / σ^2, -e / σ^2, (-2g / (2π) + 2e) / (ρ2 * σ^2))
end

"""Sample a windowed kernel (ncomp components) on the periodic grid and FFT it (zero where the window is below
`wmin`, without calling `kfun` there)."""
function _windowed_fft(kfun, ncomp::Int, M, P, L, rc, σ; wmin::Real=0.0)
    hx = P[1] / M[1]; hy = P[2] / M[2]
    K = zeros(ComplexF64, M[1], M[2], ncomp)
    Threads.@threads for b in 0:M[2]-1
        ry = (b < M[2] ÷ 2 ? b : b - M[2]) * hy
        wy = erfc((abs(ry) - L[2] - rc) / σ) / 2
        for a in 0:M[1]-1
            rx = (a < M[1] ÷ 2 ? a : a - M[1]) * hx
            wx = erfc((abs(rx) - L[1] - rc) / σ) / 2
            W = wx * wy
            W < wmin && continue
            v = kfun(Vec2(rx, ry))
            for c in 1:ncomp
                K[a+1, b+1, c] = W * v[c]
            end
        end
    end
    FFTW.fft!(K, (1, 2))
    K .* (hx * hy / (P[1] * P[2]))
end

"""
Short-range interaction matrix stored by target: column 2i−1 / 2i of `St` holds the weights of
target i's component 1 / 2 (rows = interleaved source components), so `St'` is the operator.
Built directly in compressed form (two threaded passes) and applied with a threaded loop over
targets. `paired`: columns 2i−1 and 2i have the same rows, which come in pairs (2j−1, 2j) (2×2 blocks,
how `_sr_matrix` and `area_correction` assemble them); the product then reads each row index once
for four values (a quarter less memory traffic than the plain CSC loop).
"""
struct SRMat
    St::SparseMatrixCSC{Float64,Int}
    nt::Int
    paired::Bool
end
SRMat(St::SparseMatrixCSC{Float64,Int}, nt::Int) = SRMat(St, nt, _paired(St, nt))

function _paired(St::SparseMatrixCSC, nt::Int)
    size(St, 2) == 2nt || return false
    cp = St.colptr; rv = St.rowval
    ok = trues(nt)
    Threads.@threads for i in 1:nt
        a, b, e = cp[2i-1], cp[2i], cp[2i+1]; n = b - a
        good = n == e - b && iseven(n)
        if good
            @inbounds for m in 0:2:n-1
                r = rv[a+m]
                good &= isodd(r) & (rv[a+m+1] == r + 1) & (rv[b+m] == r) & (rv[b+m+1] == r + 1)
            end
        end
        ok[i] = good
    end
    all(ok)
end

function _cell_list(src, rc)
    lo = reduce((a, b) -> min.(a, b), src)
    cell(p) = (floor(Int, (p[1] - lo[1]) / rc), floor(Int, (p[2] - lo[2]) / rc))
    cells = Dict{Tuple{Int,Int},Vector{Int}}()
    for (j, y) in enumerate(src)
        push!(get!(cells, cell(y), Int[]), j)
    end
    cells, cell
end

function _neighbors!(out::Vector{Int}, x, src, cells, cell, rc2, skip::Int)
    empty!(out)
    (ci, cj) = cell(x)
    for di in -1:1, dj in -1:1
        lst = get(cells, (ci + di, cj + dj), nothing)
        lst === nothing && continue
        for j in lst
            j == skip && continue
            r = x - src[j]; ρ2 = r[1]^2 + r[2]^2
            (ρ2 < rc2 && ρ2 > 0) && push!(out, j)
        end
    end
    sort!(out)
end

# Threaded `f(buf, i)` over 1:n in chunks, one neighbor buffer per chunk. Not `@threads :static`
# with per-thread buffers: :static errors for the rest of the session once a threaded loop has
# been interrupted (Julia leaves its threaded-region flag set).
function _chunked_foreach(f, n::Int)
    nchunk = max(1, min(n, 8 * Threads.nthreads()))
    @sync for rng in Iterators.partition(1:n, cld(max(n, 1), nchunk))
        Threads.@spawn (b = Int[]; for i in rng; f(b, i); end)
    end
end

"""
Short-range matrix for targets `T` and sources `src` (pairs closer than rc). `kfun(i, r)` returns
(K11, K12, K22) of a symmetric block, or with `full=true` (K11, K21, K12, K22).
"""
function _sr_matrix(src, T, selfmode, rc, kfun; full::Bool=false)
    cells, cell = _cell_list(src, rc)
    nt = length(T); rc2 = rc^2
    cnt = zeros(Int, nt)
    _chunked_foreach(nt) do b, i
        _neighbors!(b, T[i], src, cells, cell, rc2, selfmode ? i : 0)
        cnt[i] = length(b)
    end
    colptr = Vector{Int}(undef, 2nt + 1); colptr[1] = 1
    for i in 1:nt
        colptr[2i] = colptr[2i-1] + 2cnt[i]
        colptr[2i+1] = colptr[2i] + 2cnt[i]
    end
    nz = colptr[end] - 1
    rowval = Vector{Int}(undef, nz); nzval = Vector{Float64}(undef, nz)
    _chunked_foreach(nt) do b, i
        _neighbors!(b, T[i], src, cells, cell, rc2, selfmode ? i : 0)
        p1 = colptr[2i-1]; p2 = colptr[2i]
        x = T[i]
        for (m, j) in enumerate(b)
            v = kfun(i, x - src[j])
            k11, k21, k12, k22 = full ? (v[1], v[2], v[3], v[4]) : (v[1], v[2], v[2], v[3])
            o = 2(m - 1)
            rowval[p1+o] = 2j - 1; nzval[p1+o] = k11; rowval[p1+o+1] = 2j; nzval[p1+o+1] = k12
            rowval[p2+o] = 2j - 1; nzval[p2+o] = k21; rowval[p2+o+1] = 2j; nzval[p2+o+1] = k22
        end
    end
    SRMat(SparseMatrixCSC(2length(src), 2nt, colptr, rowval, nzval), nt)
end

SparseArrays.nnz(S::SRMat) = nnz(S.St)
Base.size(S::SRMat) = (2S.nt, size(S.St, 1))
Base.size(S::SRMat, d::Integer) = size(S)[d]
Base.Matrix(S::SRMat) = Matrix(transpose(S.St))
"""Dense block S[I, J] (cost ∝ nonzeros in the rows I, independent of the matrix size)."""
function Base.getindex(S::SRMat, I::AbstractVector{<:Integer}, J::AbstractVector{<:Integer})
    St = S.St; rv = rowvals(St); nv = nonzeros(St)
    pos = Dict{Int,Int}(j => b for (b, j) in enumerate(J))
    M = zeros(length(I), length(J))
    @inbounds for (a, i) in enumerate(I), q in nzrange(St, i)
        b = get(pos, rv[q], 0)
        b > 0 && (M[a, b] += nv[q])
    end
    M
end

"""y ← α S x + β y (threaded over targets, or over target components if not `paired`)."""
function LinearAlgebra.mul!(y::AbstractVector, S::SRMat, x::AbstractVector, α::Number, β::Number)
    St = S.St; cp = St.colptr; rv = St.rowval; nv = St.nzval
    if S.paired
        Threads.@threads for i in 1:S.nt
            a, b = cp[2i-1], cp[2i]
            s1 = 0.0; s2 = 0.0
            @inbounds for m in 0:2:b-a-1
                j = rv[a+m]; x1 = x[j]; x2 = x[j+1]
                s1 += nv[a+m] * x1 + nv[a+m+1] * x2
                s2 += nv[b+m] * x1 + nv[b+m+1] * x2
            end
            @inbounds if β == 0
                y[2i-1] = α * s1; y[2i] = α * s2
            else
                y[2i-1] = α * s1 + β * y[2i-1]; y[2i] = α * s2 + β * y[2i]
            end
        end
        return y
    end
    Threads.@threads for c in 1:2S.nt
        acc = 0.0
        @inbounds for p in cp[c]:cp[c+1]-1
            acc += nv[p] * x[rv[p]]
        end
        @inbounds y[c] = β == 0 ? α * acc : α * acc + β * y[c]
    end
    y
end

Base.:*(S::SRMat, x::AbstractVector{Float64}) = mul!(zeros(2S.nt), S, x, 1.0, 0.0)

"""Choose σ so that targets have about `nneigh` sources within c σ (sampled neighbor counts)."""
function _calibrate_sigma(src, T, c, nneigh, σ0)
    rc = c * σ0
    cells, cell = _cell_list(src, rc)
    # an evenly strided sample (it was rand(1:N, 2000): σ, rc and the short-range sparsity then changed from run
    # to run, and with them the results at the Ewald tolerance)
    idx = length(T) <= 2000 ? collect(1:length(T)) : round.(Int, range(1, length(T), length=2000))
    b = Int[]; n = 0
    for i in idx
        _neighbors!(b, T[i], src, cells, cell, rc^2, 0)
        n += length(b)
    end
    n0 = max(n / length(idx), 1.0)
    σ0 * sqrt(nneigh / n0)
end

struct FilmEwald
    σ::Float64
    rc::Float64
    M::NTuple{2,Int}
    Kaa::Array{Float64,3}          # area→area velocity multiplier (2D + 3D), 3 comps (real, even)
    Kba::Array{Float64,3}          # Γ→area velocity multiplier (2D), 3 comps
    Tab::Array{ComplexF64,3}       # area→Γ stress multiplier, 6 comps (odd → imaginary)
    p1a::FINUFFT.finufft_plan{Float64}
    p1b::FINUFFT.finufft_plan{Float64}
    p2a::FINUFFT.finufft_plan{Float64}
    p2b::FINUFFT.finufft_plan{Float64}
    SRaa::SRMat                    # area → area (i ≠ j)
    SRab::SRMat                    # Γ → area
    SRba::SRMat                    # area → Γ (traction, normals contracted)
    selfcoef::Float64
    nb::Vector{Vec2}
    timings::Dict{Symbol,Float64}
    # optional: 3D velocity at Γ from the area sources (edge terms of chiral films)
    K3::Union{Nothing,Array{Float64,3}}
    p2b3::Union{Nothing,FINUFFT.finufft_plan{Float64}}
    SR3ba::Union{Nothing,SRMat}
end

"""
    FilmEwald(xa, xb, nb; μ_2D, μ_3D, remainder=nothing, tol, σ=nothing, nneigh=150)

Plan for the far-field sums of the film operator with area nodes `xa`, boundary nodes `xb`
and boundary normals `nb` (see file header). The 3D part is the half-space Stokeslet `G3` at `μ_3D`: for an unbounded fluid of viscosity μ_f, pass μ_3D = 2μ_f. A `remainder` (a tensor `SymbolKernel` with base `Stokeslet3D()`, at
μ = 1; e.g. R_H = G_H − G3 of a finite depth) is added to the 3D multipliers (as in `EwaldStokes3D`).
The area nodes must be distinct and no boundary node may sit on an area node (true of `build_mesh`, whose area nodes
are interior Gauss points): only the area self pairs i = j are skipped, so an `ArgumentError` is thrown otherwise.
"""
function FilmEwald(xa::AbstractVector{Vec2}, xb::AbstractVector{Vec2}, nb::AbstractVector{Vec2};
                   μ_2D::Real=1.0, μ_3D::Real=1.0, remainder=nothing, tol::Real=DEFAULT_TOL, σ=nothing, nneigh::Real=400,
                   nthreads::Int=Threads.nthreads(), boundary3d::Bool=false)
    tm = Dict{Symbol,Float64}()
    _smooth_remainder(remainder, "FilmEwald")
    isempty(_coincident_pairs(xa, xa, true)) && isempty(_coincident_pairs(xa, xb, false)) ||
        throw(ArgumentError("FilmEwald: duplicated area nodes, or a boundary node on an area node (the long-range " *
                            "sums would count those pairs)"))
    allp = vcat(xa, xb)
    lo = reduce((a, b) -> min.(a, b), allp); hi = reduce((a, b) -> max.(a, b), allp)
    c0 = (lo + hi) / 2
    L = max.(hi - lo, 1e-12)
    c = max(erfcinv(tol), sqrt(log(1 / tol)), 3.0)
    # (a new binding: reassigning the keyword would box it in the kernel closures below)
    σ0 = σ === nothing ? _calibrate_sigma(xa, xa, c, nneigh, sqrt(nneigh / (π * length(xa) / (L[1] * L[2]))) / c) :
                         Float64(σ)
    # remainder: the grid's Nyquist wavenumber π/h = 2(c + 1)/σ must be ≥ _NYQ_H/scale
    RH = remainder
    σs = RH === nothing ? σ0 : min(σ0, 2(c + 1) * RH.scale / _NYQ_H)
    rc = c * σs
    S = L .+ 2rc
    P = Tuple(2 .* S)
    h = π * σs / (2(c + 1))                  # margin for the polynomial factor of the 2D multiplier
    M = ntuple(d -> 2 * ceil(Int, P[d] / (2h)), 2)
    c2 = 1 / (4π * μ_2D); c3 = 1 / (4π * μ_3D)
    tm[:multipliers] = @elapsed begin
        # 3D long range, with the remainder at μ = 1 scaled by 4π (the whole is scaled by c3 = 1/(4πμ_3D))
        k3(r) = (ρ = norm(r); (φ, dφ) = _phiL(ρ, σs); k = _Kphi(r, φ, dφ, ρ);
                 RH === nothing ? k : k .+ 4π .* remainder_tensor(RH, r))
        Kaa = real.(_windowed_fft(r -> c2 .* _S2L(r, σs) .+ c3 .* k3(r), 3, M, P, L, rc, σs))
        Kba = real.(_windowed_fft(r -> c2 .* _S2L(r, σs), 3, M, P, L, rc, σs))
        Tab = _windowed_fft(r -> _T2L(r, σs), 6, M, P, L, rc, σs)
        K3 = boundary3d ? real.(_windowed_fft(r -> c3 .* k3(r), 3, M, P, L, rc, σs)) : nothing
    end
    tm[:plans] = @elapsed begin
        sc(p, d) = 2π * (p[d] - c0[d]) / P[d]
        ax = [sc(p, 1) for p in xa]; ay = [sc(p, 2) for p in xa]
        bx = [sc(p, 1) for p in xb]; by = [sc(p, 2) for p in xb]
        ntol = max(tol / 10, 100eps())     # FINUFFT's double-precision floor
        mk(t, nt, iflag) = FINUFFT.finufft_makeplan(t, [M[1], M[2]], iflag, nt, ntol; modeord=1, nthreads=nthreads)
        # the densities and the fields are real: two components go in one complex transform (`_spread`, `_packed`)
        p1a = mk(1, 1, -1); FINUFFT.finufft_setpts!(p1a, ax, ay)
        p1b = mk(1, 1, -1); FINUFFT.finufft_setpts!(p1b, bx, by)
        p2a = mk(2, 1, +1); FINUFFT.finufft_setpts!(p2a, ax, ay)
        p2b = mk(2, 1, +1); FINUFFT.finufft_setpts!(p2b, bx, by)     # σ₁₁ + i σ₁₂, then σ₂₂ (`ewald_traction`)
        p2b3 = boundary3d ? (q = mk(2, 1, +1); FINUFFT.finufft_setpts!(q, bx, by); q) : nothing
    end
    tm[:shortrange] = @elapsed begin
        SRaa = _sr_matrix(xa, xa, true, rc, (i, r) -> c2 .* _S2S(r, σs) .+ c3 .* (ρ = norm(r); (φ, dφ) = _phiS(ρ, σs); _Kphi(r, φ, dφ, ρ)))
        SRab = _sr_matrix(xb, xa, false, rc, (i, r) -> c2 .* _S2S(r, σs))
        SRba = _sr_matrix(xa, xb, false, rc, (i, r) -> begin
            t = _T2S(r, σs); n = nb[i]
            # traction t_k = Σ_l (σ⁽ˡ⁾ n)_k f_l  →  2×2 block [t₁⁽¹⁾ t₁⁽²⁾; t₂⁽¹⁾ t₂⁽²⁾] as (11, 21, 12, 22)
            (t[1] * n[1] + t[2] * n[2], t[2] * n[1] + t[3] * n[2], t[4] * n[1] + t[5] * n[2], t[5] * n[1] + t[6] * n[2])
        end; full=true)
        SR3ba = boundary3d ? _sr_matrix(xa, xb, false, rc,
            (i, r) -> c3 .* (ρ = norm(r); (φ, dφ) = _phiS(ρ, σs); _Kphi(r, φ, dφ, ρ))) : nothing
    end
    W0 = (erfc((-L[1] - rc) / σs) / 2) * (erfc((-L[2] - rc) / σs) / 2)
    selfcoef = W0 * (c2 * (1 + _γE / 2 - log(σs)) + c3 * 2 / (σs * sqrt(π)))
    FilmEwald(σs, rc, M, Kaa, Kba, Tab, p1a, p1b, p2a, p2b, SRaa, SRab, SRba, selfcoef, collect(nb), tm,
              K3, p2b3, SR3ba)
end

"""
Spectra (M₁ × M₂ × 2) of the two components of the real densities f: one type-1 transform of f₁ + i f₂ (plan p,
one density), G = F₁ + i F₂ with F₁, F₂ Hermitian, split as F₁(k) = (G(k) + conj G(−k))/2, F₂(k) = (G(k) − conj G(−k))/2i.
M is even, so the Nyquist mode −M/2 has no partner and is split as if it were its own (an error of the size of the
multipliers there, ~e^{−(c+1)²}).
"""
function _spread(p, M, f::AbstractVector{Vec2})
    M1, M2 = M
    c = Vector{ComplexF64}(undef, length(f))
    @inbounds for j in eachindex(f)
        c[j] = complex(f[j][1], f[j][2])
    end
    G = Matrix{ComplexF64}(undef, M1, M2)
    FINUFFT.finufft_exec!(p, c, G)
    F = Array{ComplexF64}(undef, M1, M2, 2)
    Threads.@threads for b in 1:M2
        bp = b == 1 ? 1 : M2 + 2 - b                 # the mode −k (modeord = 1)
        @inbounds for a in 1:M1
            ap = a == 1 ? 1 : M1 + 2 - a
            g = G[a, b]; h = conj(G[ap, bp])
            F[a, b, 1] = 0.5 * (g + h)
            F[a, b, 2] = -0.5im * (g - h)
        end
    end
    F
end

"""Type-2 transform (plan p, one density) of V₁ + i V₂, the spectra of two real fields: their values at the plan's
points, as (V₁ field) + i (V₂ field)."""
function _packed(p, W::Matrix{ComplexF64}, n::Int)
    U = Vector{ComplexF64}(undef, n)
    FINUFFT.finufft_exec!(p, W, U)
    U
end

"""Type-1 NUFFT of the area sources (shared by `ewald_velocity` and `ewald_traction`)."""
ewald_spread(E::FilmEwald, fa::AbstractVector{Vec2}) = _spread(E.p1a, E.M, fa)

"""Σ_area [G2+G3] fa (self excluded) + Σ_Γ G2 fb at the area nodes; `Fa = ewald_spread(E, fa)`."""
function ewald_velocity(E::FilmEwald, Fa::Array{ComplexF64,3}, fa::AbstractVector{Vec2}, fb::AbstractVector{Vec2})
    M1, M2 = E.M
    Fb = _spread(E.p1b, E.M, fb)
    W = Matrix{ComplexF64}(undef, M1, M2)
    Ka, Kb = E.Kaa, E.Kba
    Threads.@threads for b in 1:M2
        @inbounds for a in 1:M1
            a1 = Fa[a, b, 1]; a2 = Fa[a, b, 2]; b1 = Fb[a, b, 1]; b2 = Fb[a, b, 2]
            v1 = Ka[a, b, 1] * a1 + Ka[a, b, 2] * a2 + Kb[a, b, 1] * b1 + Kb[a, b, 2] * b2
            v2 = Ka[a, b, 2] * a1 + Ka[a, b, 3] * a2 + Kb[a, b, 2] * b1 + Kb[a, b, 3] * b2
            W[a, b] = v1 + im * v2
        end
    end
    Ua = _packed(E.p2a, W, length(fa))
    usr = E.SRaa * interleave(fa) .+ E.SRab * interleave(fb)
    u = Vector{Vec2}(undef, length(fa))
    @inbounds for i in eachindex(u)
        u[i] = Vec2(real(Ua[i]) + usr[2i-1], imag(Ua[i]) + usr[2i]) - E.selfcoef * fa[i]
    end
    u
end

"""Σ_Γ G2 fb at the area nodes (`ewald_velocity` with fa = 0)."""
function ewald_velocity_boundary(E::FilmEwald, fb::AbstractVector{Vec2})
    M1, M2 = E.M
    Fb = _spread(E.p1b, E.M, fb)
    W = Matrix{ComplexF64}(undef, M1, M2)
    Kb = E.Kba
    Threads.@threads for b in 1:M2
        @inbounds for a in 1:M1
            b1 = Fb[a, b, 1]; b2 = Fb[a, b, 2]
            W[a, b] = (Kb[a, b, 1] * b1 + Kb[a, b, 2] * b2) + im * (Kb[a, b, 2] * b1 + Kb[a, b, 3] * b2)
        end
    end
    na = E.SRab.nt
    Ua = _packed(E.p2a, W, na)
    usr = E.SRab * interleave(fb)
    [Vec2(real(Ua[i]) + usr[2i-1], imag(Ua[i]) + usr[2i]) for i in 1:na]
end

"""Interior traction at Γ of Σ_area G2 fa (normals of the plan); `Fa = ewald_spread(E, fa)`. The stress
fields are real (odd imaginary multipliers of Hermitian spectra), so σ₁₁ + i σ₁₂ and σ₂₂ go in two
transforms instead of three."""
function ewald_traction(E::FilmEwald, Fa::Array{ComplexF64,3}, fa::AbstractVector{Vec2})
    M1, M2 = E.M
    S1 = Matrix{ComplexF64}(undef, M1, M2); S2 = Matrix{ComplexF64}(undef, M1, M2)
    T = E.Tab
    Threads.@threads for b in 1:M2
        @inbounds for a in 1:M1
            a1 = Fa[a, b, 1]; a2 = Fa[a, b, 2]
            S1[a, b] = (T[a, b, 1] * a1 + T[a, b, 4] * a2) + im * (T[a, b, 2] * a1 + T[a, b, 5] * a2)
            S2[a, b] = T[a, b, 3] * a1 + T[a, b, 6] * a2
        end
    end
    nbn = length(E.nb)
    # two single transforms: on the dumbbell (2040 × 776 grid) faster than one batched call of two
    Sb1 = _packed(E.p2b, S1, nbn); Sb2 = _packed(E.p2b, S2, nbn)
    tsr = E.SRba * interleave(fa)
    t = Vector{Vec2}(undef, nbn)
    @inbounds for i in 1:nbn
        n = E.nb[i]
        s11 = real(Sb1[i]); s12 = imag(Sb1[i]); s22 = real(Sb2[i])
        t[i] = Vec2(s11 * n[1] + s12 * n[2] + tsr[2i-1], s12 * n[1] + s22 * n[2] + tsr[2i])
    end
    t
end

"""3D in-plane Stokeslet sum Σ_area G3 fa at the Γ nodes (needs `boundary3d=true`)."""
function ewald_velocity3d_boundary(E::FilmEwald, Fa::Array{ComplexF64,3}, fa::AbstractVector{Vec2})
    E.K3 === nothing && error("FilmEwald was built without boundary3d")
    M1, M2 = E.M
    W = Matrix{ComplexF64}(undef, M1, M2)
    K = E.K3
    Threads.@threads for b in 1:M2
        @inbounds for a in 1:M1
            a1 = Fa[a, b, 1]; a2 = Fa[a, b, 2]
            W[a, b] = (K[a, b, 1] * a1 + K[a, b, 2] * a2) + im * (K[a, b, 2] * a1 + K[a, b, 3] * a2)
        end
    end
    nbn = length(E.nb)
    Ub = _packed(E.p2b3, W, nbn)
    usr = E.SR3ba * interleave(fa)
    [Vec2(real(Ub[i]) + usr[2i-1], imag(Ub[i]) + usr[2i]) for i in 1:nbn]
end

"""
    film_ewald_apply(E, fa, fb) -> (u_area, t_Γ)

u_area = Σ_area [G2+G3] fa (self excluded) + Σ_Γ G2 fb at the area nodes, and the interior
traction at Γ of the 2D Stokeslet sum over the area sources (fa, fb include quadrature weights).
"""
function film_ewald_apply(E::FilmEwald, fa::AbstractVector{Vec2}, fb::AbstractVector{Vec2})
    Fa = ewald_spread(E, fa)
    ewald_velocity(E, Fa, fa, fb), ewald_traction(E, Fa, fa)
end
