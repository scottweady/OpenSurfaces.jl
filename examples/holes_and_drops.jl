# A film of two drops, one with two holes: discretize the curves together (`discretize_curves`, which refines panels
# facing a nearby curve), mesh them (`build_mesh` on the drops `[[outer, hole₁, hole₂], [drop₂]]`, holes clockwise
# by `reverse_curve`), report the mesh quality, and integrate on the mesh's panels (Gauss–Legendre in s and t through
# `panel_map`): the area against the exact area, and the second moment ∫ |x − c|² dA of the round drop against π R⁴/2.
# (The mesh's own weights `ad.w` carry the edge weight (1 + t)^{-1/2} on the edge panels: they integrate densities
# with a d^{-1/2} edge singularity, not smooth ones; a smooth density goes through `Evaluator(...; density=:smooth)`.)
#   julia -t 8 --project=. examples/holes_and_drops.jl          (from the package root)
using OpenSurfaces, LinearAlgebra, Printf

p = 10
a, b = 1.6, 1.0                                                  # the outer ellipse
h1, r1 = Vec2(-0.6, 0.15), 0.35                                  # the holes (center, radius)
h2, r2 = Vec2(0.5, -0.25), 0.3
c2, R = Vec2(2.6, 0.3), 0.6                                      # the second drop
circ(c, r) = θ -> c + r * Vec2(cos(θ), sin(θ))
outer, hole1, hole2 = discretize_curves([θ -> Vec2(a * cos(θ), b * sin(θ)), circ(h1, r1), circ(h2, r2)], p; tol=1e-10)
drop2 = discretize_curve(circ(c2, R), p; tol=1e-10)
bd, ad, _ = build_mesh([[outer, reverse_curve(hole1), reverse_curve(hole2)], [drop2]], p)
q = mesh_quality(ad)
@printf("%d drops, %d curves, %d boundary panels; %d area panels, %d nodes; min J ratio %.2f, max aspect ratio %.1f (interior %.1f)\n",
        ndrops(bd), ncurves(bd), npanels(bd), npanels(ad), length(ad), q.jratio_min,
        q.aspect_max, q.aspect_max_interior)
g = gl_rule(p)
integrate(f, ks) = sum(begin
    x, _, _, J = panel_map(ad.panels[k], g.x[i], g.x[j]); g.w[i] * g.w[j] * J * f(x)
end for k in ks for i in 1:p, j in 1:p)
A = integrate(x -> 1.0, 1:npanels(ad))
Aex = π * (a * b - r1^2 - r2^2 + R^2)
@printf("area %.14f, exact %.14f, error %.1e\n", A, Aex, abs(A - Aex))
M2 = integrate(x -> norm(x - c2)^2, drop_panels(ad, 2))
@printf("second moment of drop 2: %.14f, exact %.14f, error %.1e\n", M2, π * R^4 / 2, abs(M2 - π * R^4 / 2))
