# The 3D Laplace single layer S_Ω[σ](x) = ∫_Ω σ(y)/(4π|x − y|) dA_y of the unit density σ = 1 on the unit disc,
# against its closed form S_Ω[1] = E(r²)/π (E the complete elliptic integral of the second kind, parameter m = r²),
# at the area nodes and at the boundary nodes on the edge Γ (where S_Ω[1] = E(1)/π = 1/π), at three tolerances.
#   julia -t 8 --project=. examples/single_layer_disc.jl          (from the package root)
using OpenSurfaces, LinearAlgebra, Printf
using SpecialFunctions: ellipe

# the unit disc: boundary panels to tol 1e-10, then the area mesh (weighted edge layer + graded layers + interior)
pans = discretize_curve(θ -> Vec2(cos(θ), sin(θ)), 10; tol=1e-10)
bd, ad, _ = build_mesh(pans, 10)
r = norm.(ad.x)
@printf("unit disc: %d boundary panels, %d area panels, %d area nodes\n", length(pans), npanels(ad), length(ad))
for tol in (1e-4, 1e-6, 1e-8)
    S = SingleLayer3D(ad; tol)                                      # S_Ω at the area nodes
    e = maximum(abs.(single_layer(S, ones(length(ad))) .- ellipe.(r .^ 2) ./ π))
    # the same potential at the boundary nodes (targets on Γ), physical (smooth) density
    EΓ = Evaluator(LaplaceKernel3D(), ad; density=:smooth, targets=bd.x, tol)
    eΓ = maximum(abs.(EΓ(ones(length(ad))) .- 1 / π))
    @printf("tol %.0e: max error at the area nodes %.1e, on Γ %.1e\n", tol, e, eΓ)
end
