# Point kernels for the film (2D Stokes) and bulk (in-plane block of 3D Stokes) problems.
#
# Normalizations, r = x − y ∈ R², ρ = |r|:
#   G3(r)       = (1/(4π μ_3D)) (I/ρ + r rᵀ/ρ³)           in-plane block of the half-space Stokeslet
#   G2(r)       = (1/(4π μ_2D)) (−log ρ I + r rᵀ/ρ²)       2D Stokeslet, log length scale 1
#   P2(r)⋅f     = (1/(2π)) (r⋅f)/ρ²                        2D Stokeslet pressure
#   σ2(r)[f]    = −(1/π) r rᵀ (r⋅f)/ρ⁴                     2D Stokeslet stress (μ_2D-independent)
#   T2(r,n) f   = σ2(r)[f] n = −(1/π) r (r⋅n)(r⋅f)/ρ⁴      traction at target with normal n
#
# G3 is the in-plane block of the Stokeslet of a half-space of viscosity μ_3D below a free
# surface (the full-space Stokeslet of viscosity μ_3D/2). All kernels are singular at r = 0;
# callers must exclude coincident points.

"""
    G3(r; μ_3D=1)

In-plane (2×2) block of the half-space Stokeslet on its free surface z = 0, (1/(4π μ_3D)) (I/ρ + r rᵀ/ρ³): a fluid of
viscosity μ_3D below the surface. It equals the full-space Stokeslet at viscosity μ_3D/2; for an unbounded fluid of viscosity μ_f, pass μ_3D = 2μ_f.
"""
@inline function G3(r::Vec2; μ_3D::Real=1.0)
    ρ2 = r[1]^2 + r[2]^2
    ρ = sqrt(ρ2)
    c = 1 / (4π * μ_3D)
    a = c / ρ
    b = c / (ρ2 * ρ)
    o = b * r[1] * r[2]
    Mat2(a + b * r[1] * r[1], o, o, a + b * r[2] * r[2])
end

"""2D Stokeslet: (1/(4π μ_2D)) (−log ρ I + r rᵀ/ρ²)."""
@inline function G2(r::Vec2; μ_2D::Real=1.0)
    ρ2 = r[1]^2 + r[2]^2
    c = 1 / (4π * μ_2D)
    a = -c * 0.5 * log(ρ2)
    b = c / ρ2
    o = b * r[1] * r[2]
    Mat2(a + b * r[1] * r[1], o, o, a + b * r[2] * r[2])
end

"""2D Stokeslet pressure row: pressure = P2(r) ⋅ f = (1/(2π)) (r⋅f)/ρ²."""
@inline function P2(r::Vec2)
    ρ2 = r[1]^2 + r[2]^2
    r / (2π * ρ2)
end

"""2D Stokeslet stress tensor σ2(r)[f] = −(1/π) r rᵀ (r⋅f)/ρ⁴ (independent of μ_2D)."""
@inline function stress2(r::Vec2, f::Vec2)
    ρ2 = r[1]^2 + r[2]^2
    c = -(r[1] * f[1] + r[2] * f[2]) / (π * ρ2 * ρ2)
    o = c * r[1] * r[2]
    Mat2(c * r[1] * r[1], o, o, c * r[2] * r[2])
end

"""
2D traction kernel: the 2×2 matrix mapping a point force f at the source to the traction
σ2(r)[f] n at the target with normal n, i.e. −(1/π) r (r⋅n) rᵀ/ρ⁴.
"""
@inline function T2(r::Vec2, n::Vec2)
    ρ2 = r[1]^2 + r[2]^2
    c = -(r[1] * n[1] + r[2] * n[2]) / (π * ρ2 * ρ2)
    o = c * r[1] * r[2]
    Mat2(c * r[1] * r[1], o, o, c * r[2] * r[2])
end

# Kernels passed to the area quadrature as callable structs, K(r) and K(i, r) (i the target): the
# quadrature is specialized on the kernel's type, and an anonymous closure is a new type at every call
# site, so the same kernel written as closures at eight sites was compiled eight times (precompile time).

"""Single-layer kernel G2(r; μ_2D) + G3(r; μ_3D); an infinite viscosity drops that term. G3 is the half-space
Stokeslet (`G3`); for an unbounded fluid of viscosity μ_f, pass μ_3D = 2μ_f. An `Evaluator` sums it by Ewald
(`EwaldStokes3D` with `μ_2D`)."""
struct StokesKernel
    μ_2D::Float64
    μ_3D::Float64
end
@inline function (K::StokesKernel)(r::Vec2)
    G = zero(Mat2)
    isfinite(K.μ_2D) && (G += G2(r; μ_2D=K.μ_2D))
    isfinite(K.μ_3D) && (G += G3(r; μ_3D=K.μ_3D))
    G
end
@inline (K::StokesKernel)(i::Int, r::Vec2) = K(r)

"""Traction kernel T2(r, n[i]) at target i."""
struct TractionKernel
    n::Vector{Vec2}
end
@inline (K::TractionKernel)(i::Int, r::Vec2) = T2(r, K.n[i])

"""`keep` predicate of `area_correction` for several drops: target i and panel k on different drops."""
struct OtherDrop
    target_drop::Vector{Int}
    panel_drop::Vector{Int}
end
(k::OtherDrop)(i::Int, j::Int) = k.target_drop[i] != k.panel_drop[j]
