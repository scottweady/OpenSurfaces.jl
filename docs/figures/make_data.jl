# The data behind the figures of OpenSurfaces' documentation, as CSV files in docs/figures/data/ (plot_figures.py
# reads them). Every number is measured here against an exact reference or a direct sum (about 2.5 minutes in all,
# 8 threads).
#   julia -t 8 --project=docs docs/figures/make_data.jl          (from the package root)
using OpenSurfaces, LinearAlgebra, Printf, Random, Logging
using OpenSurfaces: SpecialFunctions

const DATA = joinpath(@__DIR__, "data")
mkpath(DATA)
cell(v) = v isa AbstractString ? "\"" * v * "\"" : string(v)      # labels quoted (they hold commas)
csv(name, header, rows) = open(joinpath(DATA, name), "w") do io
    println(io, join(header, ","))
    for r in rows
        println(io, join(cell.(r), ","))
    end
    println("wrote data/", name, " (", length(rows), " rows)")
end
relmax(u, v) = maximum(norm.(u .- v)) / maximum(norm.(v))
circle = θ -> Vec2(cos(θ), sin(θ))
circ(c, r) = θ -> c + r * Vec2(cos(θ), sin(θ))

# ---- gallery: a film of two drops, one with two holes (as examples/holes_and_drops.jl) ---------------------------
t0 = time()
p = 10
a, b = 1.6, 1.0
h1, r1 = Vec2(-0.6, 0.15), 0.35
h2, r2 = Vec2(0.5, -0.25), 0.3
c2, R = Vec2(2.6, 0.3), 0.6
outer, hole1, hole2 = discretize_curves([θ -> Vec2(a * cos(θ), b * sin(θ)), circ(h1, r1), circ(h2, r2)], p; tol=1e-10)
drop2 = discretize_curve(circ(c2, R), p; tol=1e-10)
bd, ad, _ = build_mesh([[outer, reverse_curve(hole1), reverse_curve(hole2)], [drop2]], p)
g = gl_rule(p)
A = sum(begin x, _, _, J = panel_map(P, g.x[i], g.x[j]); g.w[i] * g.w[j] * J end for P in ad.panels for i in 1:p, j in 1:p)
@printf("gallery mesh: %d boundary panels, %d area panels, %d nodes, area error %.1e\n", npanels(bd), npanels(ad),
        length(ad), abs(A - π * (a * b - r1^2 - r2^2 + R^2)))
# panel outlines (counterclockwise in (s, t), 12 points a side) with their class
role(P) = P.weighted ? "weighted" : P.bpanel > 0 ? "graded" : "interior"
side = range(-1, 1, length=12)
rows = []
for (k, P) in enumerate(ad.panels)
    st = [[(s, -1.0) for s in side]; [(1.0, t) for t in side]; [(s, 1.0) for s in reverse(side)]; [(-1.0, t) for t in reverse(side)]]
    for (s, t) in st
        x = panel_map(P, s, t)[1]
        push!(rows, (k, role(P), x[1], x[2]))
    end
end
csv("gallery_panels.csv", ("panel", "class", "x", "y"), rows)
# the nodes near the tip of the outer ellipse, for the zoom on the edge layers
csv("gallery_nodes.csv", ("x", "y", "class"), [(x[1], x[2], role(ad.panels[(i - 1) ÷ p^2 + 1])) for (i, x) in enumerate(ad.x)
                                               if 1.40 < x[1] && abs(x[2]) < 0.1])
csv("gallery_curves.csv", ("curve", "x", "y"),
    [(c, x[1], x[2]) for (c, pans) in enumerate((outer, hole1, hole2, drop2)) for P in pans for s in range(-1, 1, length=16)
     for x in (curve_eval(P, s)[1],)])
# the 3D Laplace single layer of a unit density on the film, interpolated to a grid (NaN outside the film)
u = single_layer(SingleLayer3D(ad), ones(length(ad)))
xs = range(-1.65, 3.25, length=490); ys = range(-1.05, 1.05, length=211)
pts = [Vec2(x, y) for y in ys for x in xs]
ug = interpolate_nodal(ad, u, pts; outside=NaN)
csv("gallery_field.csv", ("x", "y", "u"), [(q[1], q[2], v) for (q, v) in zip(pts, ug)])
@printf("gallery: S[1] in [%.4f, %.4f] (%.0f s)\n", minimum(u), maximum(u), time() - t0)

# ---- convergence (a): area quadrature on the unit disc against closed forms, in p ---------------------------------
t0 = time()
rows = []
for p in (6, 8, 10, 12, 14)
    local bd, ad
    pans = discretize_curve(circle, p; tol=1.0, nstart=16, lfs_factor=Inf)     # 16 panels whatever p
    bd, ad, _ = build_mesh(pans, p)
    r = norm.(ad.x); tol = 1e-10
    push!(rows, (p, "S[1] = E(r²)/π", maximum(abs.(single_layer(SingleLayer3D(ad; tol), ones(length(ad))) .-
                                                    SpecialFunctions.ellipe.(r .^ 2) ./ π))))
    f = (1 ./ sqrt.(max.(1 .- r .^ 2, 0))) ./ ad.wfac                         # 1/√(1 − r²), stored density
    push!(rows, (p, "S[1/ω] = π/4", maximum(abs.(Evaluator(LaplaceKernel3D(), ad; tol)(f) .- π / 4))))
    push!(rows, (p, "S[1/ω] = π/4 on Γ", maximum(abs.(Evaluator(LaplaceKernel3D(), ad; targets=bd.x, tol)(f) .- π / 4))))
    F = [Vec2(1 / (2π * sqrt(max(1 - norm(x)^2, 0))), 0) / w for (x, w) in zip(ad.x, ad.wfac)]
    push!(rows, (p, "Stokeslet, rigid disc", maximum(norm.(Evaluator(Stokeslet3D(; μ_3D=2.0), ad; tol)(F) .- Ref(Vec2(3 / 32, 0))))))
end
csv("conv_area.csv", ("p", "case", "error"), rows)
@printf("convergence (a): %.0f s\n", time() - t0)

# ---- convergence (b): boundary quadrature, on Γ and near it, in p ---------------------------------------------------
t0 = time()
rows = []; m = 5
star = θ -> (1 + 0.3cos(5θ)) * Vec2(cos(θ), sin(θ))
for p in (4, 6, 8, 10, 12, 14, 16)
    tol = 1e-12
    # the 2D Laplace single layer of cos 5θ on the unit circle (12 panels): −r⁵ cos(5θ)/10 inside and on Γ
    bc = BoundaryDisc(discretize_curve(circle, p; tol=1.0, nstart=12, lfs_factor=Inf), p)
    cm = [cos(m * atan(x[2], x[1])) for x in bc.x]
    Y = [r * Vec2(cos(θ), sin(θ)) for θ in range(0.01, 2π, length=60) for r in (0.9, 0.99, 1 - 1e-4, 1 - 1e-6, 1 - 1e-8)]
    uY = Evaluator(LaplaceKernel2D(), bc; targets=Y, tol)(cm)
    push!(rows, (p, "Laplace, near Γ", maximum(abs(uY[i] + norm(Y[i])^m * cos(m * atan(Y[i][2], Y[i][1])) / 2m) for i in eachindex(Y))))
    push!(rows, (p, "Laplace, on Γ", maximum(abs.(Evaluator(LaplaceKernel2D(), bc; targets=bc.x, tol)(cm) .+ cm ./ 2m))))
    # the 2D Stokes single layer of the normal on a star (48 panels): S[n] = 0 on Γ and inside
    bs = BoundaryDisc(discretize_curve(star, p; tol=1.0, nstart=48, lfs_factor=Inf), p)
    lq = LineQuad(bs; tol)
    push!(rows, (p, "Stokes, on Γ", maximum(abs, slp2d_velocity_matrix(lq) * interleave(bs.n))))
    X = [bs.x[i] - d * bs.n[i] for i in 1:3:length(bs) for d in (1e-1, 1e-2, 1e-4, 1e-6, 1e-8)]
    push!(rows, (p, "Stokes, near Γ", maximum(norm, slp2d_eval(lq, X, bs.n))))
end
csv("conv_line.csv", ("p", "case", "error"), rows)
@printf("convergence (b): %.0f s\n", time() - t0)

# ---- convergence (c): fast sums against direct sums, in tol --------------------------------------------------------
t0 = time()
_, adf, _ = build_mesh(discretize_curve(θ -> Vec2(1.5cos(θ), 0.75sin(θ)), 10), 10)
src = adf.x; rng = MersenneTwister(1)
f = [Vec2(randn(rng), randn(rng)) for _ in src] .* adf.w
u3 = stokes3d_plane_direct(src, f; eps=1e-14).u
u2 = stokes2d_direct(src, f; eps=1e-14).u
rows = []
for tol in (1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10, 1e-11)
    push!(rows, (tol, "3D Stokeslet, Ewald", relmax(ewald_apply(EwaldStokes3D(src; tol), f), u3)))
    push!(rows, (tol, "3D Stokeslet, FMM3D", relmax(stokes3d_plane_apply(src, f; eps=tol).u, u3)))
    push!(rows, (tol, "2D Stokeslet, FMM2D", relmax(stokes2d_apply(src, f; eps=tol).u, u2)))
end
csv("conv_fast.csv", ("tol", "method", "error"), rows)
@printf("convergence (c): %d points, %.0f s\n", length(src), time() - t0)

# ---- convergence (d): SymbolKernel tables against closed forms, in the Chebyshev points per panel ----------------
t0 = time()
H = 0.5; c = 2H; qc(ρ) = sqrt(ρ^2 + c^2)
# e^{−ck}/k ↔ 1/(2π√(ρ² + c²)), and its tensor form (0, e^{−ck}/k) on Stokeslet3D (test/common/symbol_pairs.jl)
tens(ρ) = (A0 = 1 / (4π * qc(ρ)); A2r = 1 / (4π * (qc(ρ) + c)^2 * qc(ρ)); (A0 + ρ^2 * A2r, -2A2r))
κ = 4.0
Δκ(k) = (s = sqrt(k^2 + κ^2); -κ^2 / (2k * s * (k + s)))
ℓκ(ρ) = ρ == 0 ? -κ / 4π : expm1(-κ * ρ) / (4π * ρ)
ρa = range(0, 14.9, length=600); ρκ = range(0, 7.4, length=600)
rows = []
for nq in (4, 6, 8, 10, 12, 14, 16, 20)
    K = SymbolKernel(k -> 1 / (2k) + exp(-c * k) / k; base=LaplaceKernel3D(), scale=H, outputs=(:value,), nq)
    push!(rows, (nq, "scalar, e^{−ck}/k", maximum(abs(kernel_remainder(K, ρ)[1] - 1 / (2π * qc(ρ))) for ρ in ρa) * 2π * c))
    T = SymbolKernel(k -> (1 / k, 1 / (2k) + exp(-c * k) / k); base=Stokeslet3D(), scale=H, nq)
    push!(rows, (nq, "tensor, (0, e^{−ck}/k)", maximum(begin
        (α, β), (α0, β0) = kernel_remainder(T, ρ), tens(ρ)
        max(abs(α - α0), abs(β - β0) * c^2) * 4π * c
    end for ρ in ρa)))
    S = with_logger(NullLogger()) do     # it warns that the algebraic tail may leave the table off near ρ = 0
        SymbolKernel(k -> 1 / (2sqrt(k^2 + κ^2)); base=LaplaceKernel3D(), scale=1 / κ, outputs=(:value,), remainder=Δκ, nq)
    end
    push!(rows, (nq, "screened, e^{−κρ}/(4πρ)", maximum(abs(kernel_remainder(S, ρ)[1] - ℓκ(ρ)) for ρ in ρκ) / (κ / 4π)))
end
csv("conv_symbol.csv", ("nq", "kernel", "error"), rows)
@printf("convergence (d): %.0f s\n", time() - t0)

# ---- examples: the charged disc and the rigid disc (examples/charged_disc.jl, rigid_disc.jl), the screened kernel --
t0 = time()
conv = []; dens = []
for p in (8, 10, 12)                                     # the settings of the example scripts
    local ad
    _, ad, _ = build_mesh(discretize_curve(circle, p), p; edge_layers=1)
    N = length(ad); ω = sqrt.(1 .- norm.(ad.x) .^ 2)
    σ̃ = evaluator_matrix(Evaluator(LaplaceKernel3D(), ad; fast=:direct, tol=1e-8)) \ ones(N)
    eσ = abs.(σ̃ .* ad.wfac .* ω .* (π / 4) .- 1)
    push!(conv, (p, "charged disc", "density", maximum(eσ)))
    push!(conv, (p, "charged disc", "total", abs(sum(ad.w .* σ̃) - 8) / 8))
    f̃ = deinterleave(evaluator_matrix(Evaluator(Stokeslet3D(), ad; fast=:direct, tol=1e-8)) \ interleave(fill(Vec2(1, 0), N)))
    ef = [norm(f̃[i] * ad.wfac[i] * ω[i] * (3π / 8) - Vec2(1, 0)) for i in 1:N]
    push!(conv, (p, "rigid disc", "density", maximum(ef)))
    push!(conv, (p, "rigid disc", "total", norm(sum(ad.w .* f̃) - Vec2(16 / 3, 0)) / (16 / 3)))
    p == 10 && for i in 1:N
        push!(dens, (norm(ad.x[i]), σ̃[i] * ad.wfac[i], eσ[i], f̃[i][1] * ad.wfac[i], ef[i]))
    end
end
csv("ex_disc_conv.csv", ("p", "case", "quantity", "error"), conv)
csv("ex_disc_density.csv", ("r", "sigma", "sigma_error", "fx", "f_error"), dens)
K = with_logger(NullLogger()) do
    SymbolKernel(k -> 1 / (2sqrt(k^2 + κ^2)); base=LaplaceKernel3D(), scale=1 / κ, outputs=(:value,), remainder=Δκ)
end
ρs = [0.0; 10 .^ range(-4, log10(7.4), length=300)]
csv("ex_screened.csv", ("rho", "table", "exact"), [(ρ, kernel_remainder(K, ρ)[1], ℓκ(ρ)) for ρ in ρs])
@printf("examples: %.0f s\n", time() - t0)

# ---- pointwise: the single layer of the elliptic disc's charge density, a constant (examples/ellipse_single_layer.jl)
t0 = time()
a, b = 2.0, 1.0
V = b * SpecialFunctions.ellipk(1 - b^2 / a^2) / 2
ωe(x) = sqrt(max(1 - (x[1] / a)^2 - (x[2] / b)^2, 0.0))
dist(x) = 1 - sqrt((x[1] / a)^2 + (x[2] / b)^2)           # 1 − the elliptic radius: 0 on Γ, 1 at the center
pts = [Vec2(a * r * cos(t), b * r * sin(t)) for r in (0.3, 0.9, 0.99, 1 - 1e-6) for t in range(0.1, 2π; length=40)]
maprows = Any[]; distrows = Any[]; tolrows = Any[]
for p in (8, 10, 12)
    bd, ad, _ = build_mesh(discretize_curve(θ -> Vec2(a * cos(θ), b * sin(θ)), p), p; edge_layers=1)
    σ̃ = 1 ./ ωe.(ad.x) ./ ad.wfac
    for tol in 10.0 .^ (-3:-1:-11)
        uA, uΓ, uP = Evaluator(LaplaceKernel3D(), ad; targets=(:area, :boundary, pts), bd, tol)(σ̃)
        eA, eΓ, eP = (abs.(u .- V) ./ V for u in (uA, uΓ, uP))
        append!(tolrows, [(p, tol, "area nodes", maximum(eA)), (p, tol, "nodes of Γ", maximum(eΓ)),
                          (p, tol, "points", maximum(eP))])
        p == 10 && tol in (1e-4, 1e-6, 1e-8, 1e-10) &&
            append!(distrows, [(tol, dist(x), e) for (x, e) in zip(ad.x, eA)])
    end
    if p == 10      # the map: the error on a grid of points of Ω, at a tol far below the floor set by p
        xs = range(-a, a; length=321); ys = range(-b, b; length=161)
        grid = [Vec2(x, y) for y in ys for x in xs]
        inside = [(q[1] / a)^2 + (q[2] / b)^2 < 1 - 1e-9 for q in grid]
        tg = @elapsed (ug = Evaluator(LaplaceKernel3D(), ad; targets=grid[inside], tol=1e-10)(σ̃))
        eg = fill(NaN, length(grid)); eg[inside] = abs.(ug .- V) ./ V
        append!(maprows, [(q[1], q[2], e) for (q, e) in zip(grid, eg)])
        outline(P, n) = [[panel_map(P, s, -1.0)[1] for s in range(-1, 1; length=n)];
                      [panel_map(P, 1.0, t)[1] for t in range(-1, 1; length=n)];
                      [panel_map(P, s, 1.0)[1] for s in range(1, -1; length=n)];
                      [panel_map(P, -1.0, t)[1] for t in range(1, -1; length=n)]]
        csv("pointwise_panels.csv", ("panel", "x", "y"), [(k, x[1], x[2]) for (k, P) in enumerate(ad.panels) for x in outline(P, 12)])
        @printf("pointwise map: %d points in %.0f s, largest error %.1e\n", count(inside), tg, maximum(filter(!isnan, eg)))
    end
end
csv("pointwise_map.csv", ("x", "y", "error"), maprows)
csv("pointwise_dist.csv", ("tol", "d", "error"), distrows)
csv("pointwise_tol.csv", ("p", "tol", "set", "error"), tolrows)
@printf("pointwise: %.0f s\n", time() - t0)
