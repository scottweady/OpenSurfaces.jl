# Fast (FMM) and direct evaluation of 2D Stokeslet sums and in-plane 3D Stokeslet sums.
#
# Library normalizations (determined empirically against direct sums, FMM2D v0.2.x,
# FMM3D v2.x; see src/OpenSurfaces/test/test_kernels.jl, which re-checks all of these):
#
#   FMM2D.stfmm2d, Stokeslet strengths σ (r = x − y, ρ = |r|):
#       U_lib(x) = Σ [ r rᵀ/(2ρ²) − (log ρ / 2) I ] σ        (natural log, scale 1, no constant)
#       P_lib(x) = Σ (r⋅σ)/ρ²
#       grad_lib[a,b,k] = ∂_a U_lib_b  at target k  (NOTE: derivative index FIRST, i.e. the
#                         transpose of the Jacobian J[i,j] = ∂_j u_i)
#     Hence, with our G2 = (1/(4π μ_2D))(−log ρ I + r rᵀ/ρ²) and P2⋅f = (1/2π)(r⋅f)/ρ²:
#       u   = U_lib / (2π μ_2D),   p = P_lib / (2π),   ∇u (J[i,j] = ∂_j u_i) = grad_libᵀ/(2π μ_2D)
#       σ   = −p I + μ_2D(J + Jᵀ) = (1/2π) [ −P_lib I + grad_lib + grad_libᵀ ]   (μ_2D drops out)
#     The log term carries no hidden constant: the library's log length scale is 1, verified
#     also for point clouds of diameter ~1e-3 and ~1e3 (where the FMM's internal box rescaling
#     would expose any missing log(scale) Σσ term).
#
#   FMM3D.stfmm3d, Stokeslet strengths σ:
#       U_lib(x) = Σ (1/(8π)) [ I/ρ + r rᵀ/ρ³ ] σ
#     Hence our G3 = (1/(4π μ_3D))(I/ρ + r rᵀ/ρ³) gives u = 2U_lib / μ_3D.
#     With z = 0 points and in-plane forces, the z-component of U_lib is exactly 0.
#
#   Self/coincident points:
#     * FMM3D drops every coincident source–target pair, both at the sources (ppreg) and at
#       separate targets (ppregt).
#     * FMM2D drops the self pair i == j at the sources (ppreg) exactly. BUT for any OTHER
#       coincident pair (explicit targets equal to a source point, or duplicated source
#       points), FMM2D drops the log term and the pressure/gradient contributions, yet still
#       adds a spurious (1/4)σ to U_lib (the angular average I/2 of r rᵀ/ρ², times 1/2).
#       In our units that is f/(8π μ_2D). `stokes2d_apply` detects exactly coincident pairs
#       (hash lookup, O(N + M)) and subtracts it. The traction needs no correction.
#     The direct routines below skip pairs with r == 0 exactly, so after this fix both agree.

# ---------------------------------------------------------------------------------------
# Interleaved layout helpers
# ---------------------------------------------------------------------------------------

"""Interleave a vector of Vec2 into [v₁₁, v₁₂, v₂₁, v₂₂, …] (length 2N)."""
interleave(v::AbstractVector{Vec2}) = collect(reinterpret(Float64, v))

"""Inverse of `interleave`: length-2N Float64 vector → Vector{Vec2}."""
function deinterleave(x::AbstractVector{<:Real})
    iseven(length(x)) || throw(ArgumentError("deinterleave: length must be even"))
    [Vec2(x[2i-1], x[2i]) for i in 1:length(x)÷2]
end

"""Vector{Vec2} → 2×N Array{Float64} (fresh copy, as required by the Fortran libraries)."""
_to2xN(v::AbstractVector{Vec2}) = reshape(collect(reinterpret(Float64, v)), 2, length(v))

function _to3xN(v::AbstractVector{Vec2})
    A = zeros(3, length(v))
    @inbounds for i in eachindex(v)
        A[1, i] = v[i][1]; A[2, i] = v[i][2]
    end
    A
end

_fromNx(A::AbstractMatrix) = [Vec2(A[1, i], A[2, i]) for i in axes(A, 2)]

# ---------------------------------------------------------------------------------------
# 2D Stokes (film)
# ---------------------------------------------------------------------------------------

"""
    stokes2d_apply(src::Vector{Vec2}, f::Vector{Vec2}, trg=nothing;
                   μ_2D=1.0, eps=DEFAULT_TOL, traction_normals=nothing) -> (u=..., t=...)

FMM evaluation of uᵢ = Σⱼ G2(xᵢ − yⱼ) fⱼ at the targets `trg` (or at the sources, with the
self term excluded, if `trg === nothing`). If `traction_normals` (one Vec2 per target) is
given, also returns tᵢ = Σⱼ σ2(xᵢ − yⱼ)[fⱼ] nᵢ (otherwise `t === nothing`).
Coincident source/target pairs are skipped.

A method taking an interleaved `f::Vector{Float64}` (length 2N) returns interleaved `u`, `t`.
"""
function stokes2d_apply(src::AbstractVector{Vec2}, f::AbstractVector{Vec2}, trg=nothing;
                        μ_2D::Real=1.0, eps::Real=DEFAULT_TOL, traction_normals=nothing)
    length(f) == length(src) || throw(DimensionMismatch("length(f) != length(src)"))
    ntrg = trg === nothing ? length(src) : length(trg)
    wantt = traction_normals !== nothing
    wantt && length(traction_normals) != ntrg &&
        throw(DimensionMismatch("traction_normals must have one entry per target"))
    if ntrg == 0 || isempty(src)
        z = zeros(Vec2, ntrg)
        return (u=z, t=wantt ? copy(z) : nothing)
    end
    flag = wantt ? 3 : 1
    S = _to2xN(src); F = _to2xN(f)
    if trg === nothing
        v = FMM2D.stfmm2d(eps=Float64(eps), sources=S, stoklet=F, ppreg=flag)
        U, Pr, Gr = v.pot, v.pre, v.grad
    else
        v = FMM2D.stfmm2d(eps=Float64(eps), sources=S, stoklet=F, targets=_to2xN(trg), ppregt=flag)
        U, Pr, Gr = v.pottarg, v.pretarg, v.gradtarg
    end
    v.ier == 0 || error("FMM2D.stfmm2d returned ier = $(v.ier)")
    cu = 1 / (2π * μ_2D)
    u = [Vec2(cu * U[1, i], cu * U[2, i]) for i in 1:ntrg]
    _fix_coincident_2d!(u, src, f, trg, 1 / (8π * μ_2D))
    t = nothing
    if wantt
        t = Vector{Vec2}(undef, ntrg)
        c = 1 / (2π)
        @inbounds for i in 1:ntrg
            n = traction_normals[i]
            # σ = c(−P I + G + Gᵀ), G[a,b] = Gr[a,b,i] (symmetric part only matters)
            s11 = c * (-Pr[i] + 2Gr[1, 1, i])
            s22 = c * (-Pr[i] + 2Gr[2, 2, i])
            s12 = c * (Gr[1, 2, i] + Gr[2, 1, i])
            t[i] = Vec2(s11 * n[1] + s12 * n[2], s12 * n[1] + s22 * n[2])
        end
    end
    return (u=u, t=t)
end

# Normalize -0.0 to 0.0 so that hashing agrees with ==.
_hkey(x::Vec2) = Vec2(x[1] + 0.0, x[2] + 0.0)

# Subtract FMM2D's spurious contribution c f_j from every coincident pair (i, j), excluding
# i == j when targets are the sources (that pair is already dropped by the library).
function _fix_coincident_2d!(u, src, f, trg, c)
    if trg === nothing
        d = Dict{Vec2,Vector{Int}}()
        @inbounds for j in eachindex(src)
            push!(get!(() -> Int[], d, _hkey(src[j])), j)
        end
        for js in values(d)
            length(js) > 1 || continue
            fs = sum(f[j] for j in js)
            for i in js
                u[i] -= c * (fs - f[i])
            end
        end
    else
        d = Dict{Vec2,Vec2}()   # point → total force of the sources located there
        @inbounds for j in eachindex(src)
            k = _hkey(src[j])
            d[k] = get(d, k, zero(Vec2)) + f[j]
        end
        @inbounds for i in eachindex(trg)
            fs = get(d, _hkey(trg[i]), nothing)
            fs === nothing || (u[i] -= c * fs)
        end
    end
    u
end

"""
    stokes2d_direct(src, f, trg=nothing; μ_2D=1.0, eps=DEFAULT_TOL, traction_normals=nothing)

Direct O(NM) version of `stokes2d_apply` (same signature and return value; `eps` ignored).
"""
function stokes2d_direct(src::AbstractVector{Vec2}, f::AbstractVector{Vec2}, trg=nothing;
                         μ_2D::Real=1.0, eps::Real=DEFAULT_TOL, traction_normals=nothing)
    length(f) == length(src) || throw(DimensionMismatch("length(f) != length(src)"))
    X = trg === nothing ? src : trg
    ntrg = length(X)
    wantt = traction_normals !== nothing
    wantt && length(traction_normals) != ntrg &&
        throw(DimensionMismatch("traction_normals must have one entry per target"))
    u = zeros(Vec2, ntrg)
    t = wantt ? zeros(Vec2, ntrg) : nothing
    Threads.@threads for i in 1:ntrg
        x = X[i]
        acc = zero(Vec2); acct = zero(Vec2)
        n = wantt ? traction_normals[i] : zero(Vec2)
        @inbounds for j in eachindex(src)
            r = x - src[j]
            (r[1] == 0 && r[2] == 0) && continue
            acc += G2(r; μ_2D=μ_2D) * f[j]
            wantt && (acct += T2(r, n) * f[j])
        end
        u[i] = acc
        wantt && (t[i] = acct)
    end
    return (u=u, t=t)
end

# ---------------------------------------------------------------------------------------
# In-plane 3D Stokes (bulk)
# ---------------------------------------------------------------------------------------

"""
    stokes3d_plane_apply(src::Vector{Vec2}, f::Vector{Vec2}, trg=nothing; μ_3D=1.0, eps=DEFAULT_TOL)
        -> (u=..., uz_max=...)

FMM evaluation of uᵢ = Σⱼ G3(xᵢ − yⱼ) fⱼ for points in the plane z = 0 and in-plane forces
(the in-plane block of the half-space Stokeslet `G3`, viscosity `μ_3D` below the surface; for an unbounded fluid of viscosity μ_f, pass μ_3D = 2μ_f).
Targets = sources with self excluded if
`trg === nothing`. `uz_max` is the max |z-velocity| reported by the FMM (should be ~0).

A method taking an interleaved `f::Vector{Float64}` returns interleaved `u`.
"""
function stokes3d_plane_apply(src::AbstractVector{Vec2}, f::AbstractVector{Vec2}, trg=nothing;
                              μ_3D::Real=1.0, eps::Real=DEFAULT_TOL)
    length(f) == length(src) || throw(DimensionMismatch("length(f) != length(src)"))
    ntrg = trg === nothing ? length(src) : length(trg)
    (ntrg == 0 || isempty(src)) && return (u=zeros(Vec2, ntrg), uz_max=0.0)
    S = _to3xN(src); F = _to3xN(f)
    if trg === nothing
        v = FMM3D.stfmm3d(Float64(eps), S; stoklet=F, ppreg=1)
        U = v.pot
    else
        v = FMM3D.stfmm3d(Float64(eps), S; stoklet=F, targets=_to3xN(trg), ppregt=1)
        U = v.pottarg
    end
    v.ier == 0 || error("FMM3D.stfmm3d returned ier = $(v.ier)")
    c = 2 / μ_3D
    u = [Vec2(c * U[1, i], c * U[2, i]) for i in 1:ntrg]
    uz = c * maximum(abs, @view U[3, :])
    return (u=u, uz_max=uz)
end

"""
    stokes3d_plane_direct(src, f, trg=nothing; μ_3D=1.0, eps=DEFAULT_TOL)

Direct O(NM) version of `stokes3d_plane_apply` (same signature; `uz_max` is 0). `G3` is the half-space Stokeslet:
for an unbounded fluid of viscosity μ_f, pass μ_3D = 2μ_f.
"""
function stokes3d_plane_direct(src::AbstractVector{Vec2}, f::AbstractVector{Vec2}, trg=nothing;
                               μ_3D::Real=1.0, eps::Real=DEFAULT_TOL)
    length(f) == length(src) || throw(DimensionMismatch("length(f) != length(src)"))
    X = trg === nothing ? src : trg
    u = zeros(Vec2, length(X))
    Threads.@threads for i in eachindex(X)
        x = X[i]
        acc = zero(Vec2)
        @inbounds for j in eachindex(src)
            r = x - src[j]
            (r[1] == 0 && r[2] == 0) && continue
            acc += G3(r; μ_3D=μ_3D) * f[j]
        end
        u[i] = acc
    end
    return (u=u, uz_max=0.0)
end

# ---------------------------------------------------------------------------------------
# Interleaved-density methods (for Krylov / LinearMaps)
# ---------------------------------------------------------------------------------------

for fn in (:stokes2d_apply, :stokes2d_direct)
    @eval function $fn(src::AbstractVector{Vec2}, f::AbstractVector{Float64}, trg=nothing;
                       traction_normals=nothing, kw...)
        r = $fn(src, deinterleave(f), trg; traction_normals=traction_normals, kw...)
        return (u=interleave(r.u), t=r.t === nothing ? nothing : interleave(r.t))
    end
end
for fn in (:stokes3d_plane_apply, :stokes3d_plane_direct)
    @eval function $fn(src::AbstractVector{Vec2}, f::AbstractVector{Float64}, trg=nothing; kw...)
        r = $fn(src, deinterleave(f), trg; kw...)
        return (u=interleave(r.u), uz_max=r.uz_max)
    end
end
