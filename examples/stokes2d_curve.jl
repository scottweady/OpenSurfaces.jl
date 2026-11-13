# The 2D Stokes single layer S[ν](x) = ∮_Γ G2(x − y) ν(y) ds_y on a boundary curve Γ (viscosity μ_2D = 1), with
# the identities it satisfies exactly: on the star r = 1 + 0.3 cos 5θ, S[n] = 0 on Γ and inside (n the outward
# normal: G2 is divergence-free), and the interior traction of S[n] is n (pressure −1); on the circle of radius a,
# a constant density e_x gives u = a (1/4 − ½ log a) e_x on Γ and inside.
#   julia -t 8 --project=. examples/stokes2d_curve.jl          (from the package root)
using OpenSurfaces, LinearAlgebra, Printf

p = 10; tol = 1e-10
bd = BoundaryDisc(discretize_curve(θ -> (1 + 0.3cos(5θ)) * Vec2(cos(θ), sin(θ)), p; tol), p)
A = slp2d_velocity_matrix(bd; tol)                            # dense S on Γ (interleaved), with the log corrections
T = traction_self_matrix(bd)                                  # its interior traction on Γ
ν = interleave(bd.n)
# interior targets down to 1e-6 from Γ (along −n at every 7th node) and a few deep inside
X = [[bd.x[i] - d * bd.n[i] for i in 1:7:length(bd) for d in (1e-2, 1e-4, 1e-6)]; [Vec2(0, 0), Vec2(0.3, -0.2)]]
uX = Evaluator(OpenSurfaces.StokesKernel(1.0, Inf), bd; targets=X, tol)(bd.n)  # corrected smooth rule + FMM2D
@printf("star, %d panels, %d nodes: S[n] on Γ %.1e, inside (%d targets) %.1e; traction of S[n] − n %.1e\n",
        npanels(bd), length(bd), maximum(abs, A * ν), length(X), maximum(norm, uX), maximum(abs, T * ν - ν))
for a in (0.5, 2.0)
    bc = BoundaryDisc(discretize_curve(θ -> a * Vec2(cos(θ), sin(θ)), p; tol), p)
    one = fill(Vec2(1, 0), length(bc)); uex = Vec2(a * (0.25 - 0.5log(a)), 0)
    uΓ = deinterleave(slp2d_velocity_matrix(bc; tol) * interleave(one))
    uin = Evaluator(OpenSurfaces.StokesKernel(1.0, Inf), bc; targets=[0.9a * Vec2(cos(t), sin(t)) for t in 0:0.5:6], tol)(one)
    @printf("circle a = %.1f: constant density, on Γ %.1e, inside %.1e\n", a, maximum(norm, uΓ .- Ref(uex)),
            maximum(norm, uin .- Ref(uex)))
end
