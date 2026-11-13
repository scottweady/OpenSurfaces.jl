# The charged disc: solve the first-kind equation S_Ω[σ] = 1 on the unit disc for the 3D Laplace kernel 1/(4πρ)
# (a conducting disc at potential 1 in free space, ε = 1). Exact: σ = 4/(π√(1 − r²)), singular at the edge, and the
# total charge (the capacitance) Q = 8. The mesh stores σ̃ = σ/wfac on its weighted edge panels, wfac = (1 + t)^{-1/2}
# (Gauss–Jacobi in t), so σ̃ is smooth and a dense solve gives σ to high accuracy. One edge layer (`edge_layers=1`):
# the graded layers (the default 3) are for d^{1/2} log d terms; with a d^{-1/2} density they cost accuracy here.
#   julia -t 8 --project=. examples/charged_disc.jl          (from the package root)
using OpenSurfaces, LinearAlgebra, Printf

for p in (8, 10, 12)
    _, ad, _ = build_mesh(discretize_curve(θ -> Vec2(cos(θ), sin(θ)), p), p; edge_layers=1)
    E = Evaluator(LaplaceKernel3D(), ad; fast=:direct, tol=1e-8)     # stored (edge-weighted) density
    σ̃ = evaluator_matrix(E) \ ones(length(ad))                      # dense N × N first-kind solve
    σ = σ̃ .* ad.wfac                                                 # the physical density at the nodes
    σex = 4 ./ (π .* sqrt.(1 .- norm.(ad.x) .^ 2))
    Q = sum(ad.w .* σ̃)                                               # ∫ σ dA (the weights carry the edge weight)
    @printf("p = %2d, N = %4d: max |σ − σ_exact|/σ_exact %.1e, charge Q = %.12f, |Q − 8| %.1e\n",
            p, length(ad), maximum(abs.(σ .- σex) ./ σex), Q, abs(Q - 8))
end
