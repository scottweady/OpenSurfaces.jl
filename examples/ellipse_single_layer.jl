# Pointwise errors of the 3D Laplace single layer S_Ω[σ](x) = ∫_Ω σ(y)/(4π|x − y|) dA_y on an ellipse with semi-axes
# a = 2, b = 1. The density σ = 1/√(1 − x²/a² − y²/b²), singular at the edge, is that of a conducting elliptic disc:
# its potential is the constant V = b K(e)/2 on all of Ω (K the complete elliptic integral of the first kind,
# parameter e² = 1 − b²/a²; total charge 2πab, capacitance 4πa/K(e)). The potential is evaluated at the area nodes,
# at the nodes of Γ, and at points in Ω off the mesh, for orders p and tolerances tol; the error at every target is
# |u − V|/V. The tolerance sets the quadrature error, and p the resolution of the density on the panels: once tol is
# below that floor, only p reduces the error. One edge layer, as for the solves (examples/charged_disc.jl).
#   julia -t 8 --project=. examples/ellipse_single_layer.jl          (from the package root)
using OpenSurfaces, LinearAlgebra, Printf
using OpenSurfaces: SpecialFunctions

a, b = 2.0, 1.0
V = b * SpecialFunctions.ellipk(1 - b^2 / a^2) / 2
ω(x) = sqrt(max(1 - (x[1] / a)^2 - (x[2] / b)^2, 0.0))
pts = [Vec2(a * r * cos(t), b * r * sin(t)) for r in (0.3, 0.9, 0.99, 1 - 1e-6) for t in range(0.1, 2π; length=40)]

@printf("V = b K(e)/2 = %.15f; max relative error at the area nodes, the nodes of Γ, and %d points:\n", V, length(pts))
for p in (8, 10, 12)
    bd, ad, _ = build_mesh(discretize_curve(θ -> Vec2(a * cos(θ), b * sin(θ)), p), p; edge_layers=1)
    σ̃ = 1 ./ ω.(ad.x) ./ ad.wfac                     # the stored density: σ̃ = σ/wfac is smooth up to Γ
    for tol in (1e-4, 1e-6, 1e-8, 1e-10)
        E = Evaluator(LaplaceKernel3D(), ad; targets=(:area, :boundary, pts), bd, tol)   # one fast sum, three sets
        uA, uΓ, uP = E(σ̃)
        err(u) = maximum(abs.(u .- V)) / V
        @printf("  p = %2d (%5d nodes), tol = %.0e: area %.1e   Γ %.1e   points %.1e\n",
                p, length(ad), tol, err(uA), err(uΓ), err(uP))
    end
end
