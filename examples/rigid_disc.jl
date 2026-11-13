# A rigid disc translating in its plane: the unit disc in the free surface z = 0 of a fluid half-space of viscosity
# μ = 1 below it, moving with velocity U = e_x. Solve S[f] = U for the traction f with the half-space Stokeslet
# `Stokeslet3D()`, G = (1/(4πμ))(I/ρ + r rᵀ/ρ³) (the full-space Stokeslet at viscosity μ/2). Exact:
# f = (8/(3π√(1 − r²))) e_x and the drag F = ∫ f dA = (16/3) μ a U, i.e. (32/3) μ_f a U for a disc moving edgewise
# in an unbounded fluid of viscosity μ_f = μ/2. One edge layer, as for the charged disc (examples/charged_disc.jl).
#   julia -t 8 --project=. examples/rigid_disc.jl          (from the package root)
using OpenSurfaces, LinearAlgebra, Printf

for p in (8, 10, 12)
    _, ad, _ = build_mesh(discretize_curve(θ -> Vec2(cos(θ), sin(θ)), p), p; edge_layers=1)
    N = length(ad)
    E = Evaluator(Stokeslet3D(), ad; fast=:direct, tol=1e-8)        # stored density, a 2N × 2N operator
    f̃ = deinterleave(evaluator_matrix(E) \ interleave(fill(Vec2(1, 0), N)))
    f = f̃ .* ad.wfac
    fex = [Vec2(8 / (3π * sqrt(1 - norm(x)^2)), 0) for x in ad.x]
    F = sum(ad.w .* f̃)
    @printf("p = %2d, N = %4d: max |f − f_exact|/|f_exact| %.1e, drag F = (%.12f, %.0e), |F − 16/3| %.1e\n",
            p, N, maximum(norm.(f .- fex) ./ norm.(fex)), F[1], F[2], norm(F - Vec2(16 / 3, 0)))
end
