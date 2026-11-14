# 2D Laplace and biharmonic kernels of the volume potentials on the film (Gomez Herrera & Weady 2026),
# in the 2×2 form of the quadrature routines (r = x − y):
#   LaplaceKernel2D        (1/2π) log ρ                          scalar, both diagonal slots
#   LaplaceGradient2D      ∇ of it, (1/2π) r/ρ²                 vector, first column
#   BiharmonicGradient2D   (1/8π) r (2 log ρ + c0),  c0 = −1 + 2κ − 2 log R: ∇ of the bilaplacian kernel
#                          (ρ²/8π)(log(ρ/R) − 1 + κ)           vector, first column
# Far field: FMM2D's Laplace FMM (`Evaluator`), or `RadialEwald2D` (fast = :ewald).

"""(1/2π) log ρ, as (1/2π) log ρ · I."""
struct LaplaceKernel2D end
@inline (::LaplaceKernel2D)(r::Vec2) = (g = log(norm(r)) / 2π; Mat2(g, 0, 0, g))

"""(1/2π) r/ρ², as the first column of a `Mat2`."""
struct LaplaceGradient2D end
@inline (::LaplaceGradient2D)(r::Vec2) = (ρ2 = r[1]^2 + r[2]^2; Mat2(r[1] / (2π * ρ2), r[2] / (2π * ρ2), 0, 0))

"""(1/8π) r (2 log ρ + c0), as the first column of a `Mat2`; c0 = −1 + 2κ − 2 log R."""
struct BiharmonicGradient2D
    c0::Float64
end
BiharmonicGradient2D(; κ::Real=0.0, R::Real=1.0) = BiharmonicGradient2D(-1 + 2κ - 2log(R))
@inline function (K::BiharmonicGradient2D)(r::Vec2)
    ρ = norm(r)
    f = (ρ == 0 ? 0.0 : 2log(ρ)) + K.c0
    Mat2(r[1] * f / (8π), r[2] * f / (8π), 0, 0)
end
