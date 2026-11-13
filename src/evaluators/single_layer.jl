# 3D Laplace single layer on a flat film (e.g. the nutrient field of the growth model of Gomez Herrera & Weady 2026,
# §3.2):
#   S_Ω[σ](x) = (1/4π) ∫_Ω σ(y)/|x − y| dy.
# σ is bounded (its edge behavior comes from c, ~ d log d), so it is integrated on the unweighted
# copy of the film mesh, as in volume.jl: an `Evaluator` of `LaplaceKernel3D` (far field FMM3D, sources in z = 0;
# sparse near corrections from `area_correction`).
# A `remainder` (a scalar `SymbolKernel` with base 1/(4πρ) and output :value, e.g. the finite-depth nutrient kernel of
# symbol tanh(kH)/(2k)) whose whole kernel 1/(4πρ) + ℓ(ρ) decays exponentially: it
# is summed directly over the pairs closer than ρc (where the kernel's tail carries less than eps of its integral),
# with the deep near corrections (ℓ is smooth, so it needs none, but it has a self term where a film node is also a
# source node: ℓ(0)). A remainder whose symbol decays only algebraically (a screened kernel e^{−κρ}/(4πρ): ℓ has a cone
# at ρ = 0) takes the near corrections of the whole kernel instead (`SymbolKernel` decay, src/evaluators/evaluator.jl).

struct SingleLayer3D
    ad::AreaDisc
    E::Evaluator                         # S_Ω (or base + remainder) at the film nodes, from smooth densities
end

"""
    SingleLayer3D(ad; tol=DEFAULT_TOL, eps=tol, remainder=nothing)

S_Ω at the film nodes (`single_layer`). A `remainder` (above; e.g. a finite depth with the concentration fixed on
the floor) makes it the operator of the kernel 1/(4πρ) + ℓ(ρ). `tol` for the near corrections, `eps` for the FMM
(with a remainder: the share of the kernel's integral cut off).
"""
SingleLayer3D(ad::AreaDisc; tol::Real=DEFAULT_TOL, eps::Real=tol, remainder=nothing) =
    SingleLayer3D(ad, Evaluator(LaplaceKernel3D(), ad; density=:smooth, remainder, tol, eps))

"""S_Ω[σ] at the film nodes, for physical nodal values σ (with a remainder, the operator of 1/(4πρ) + ℓ)."""
single_layer(L::SingleLayer3D, σ::AbstractVector{<:Real}) = L.E(σ)
