# Geometry: the Gauss–Legendre and Gauss–Jacobi panel rules (src/geometry/panels.jl), boundary panels
# (src/geometry/boundary.jl), area meshes with graded edge layers
# (src/geometry/mesh.jl) with and without holes, several drops. Exact moments, areas,
# second moments and perimeters; coverage of the domain by the panels exactly once; layout of the
# edge columns; rejection of overlapping or wrongly oriented curves.
#   julia -t 8 --project -e 'using Pkg; Pkg.test(test_args=["geometry"])'   (from src/OpenSurfaces)
using OpenSurfaces, LinearAlgebra, StaticArrays, Test
using OpenSurfaces: gausslegendre

using OpenSurfaces: discretize_curve, build_mesh, mesh_quality, curve_area, curve_length, panel_length,
    local_feature_size, point_in_polygon

# ---------------------------------------------------------------------------------------
# Test domains: (γ, γ') with exact area / perimeter by the periodic trapezoidal rule
# (spectrally accurate for these analytic curves).
# ---------------------------------------------------------------------------------------
const DOMAINS = [
    "disk"     => (θ -> Vec2(cos(θ), sin(θ)),
                   θ -> Vec2(-sin(θ), cos(θ))),
    "ellipse"  => (θ -> Vec2(2cos(θ), 0.5sin(θ)),
                   θ -> Vec2(-2sin(θ), 0.5cos(θ))),
    "star"     => (θ -> (1 + 0.3cos(5θ)) * Vec2(cos(θ), sin(θ)),
                   θ -> -1.5sin(5θ) * Vec2(cos(θ), sin(θ)) + (1 + 0.3cos(5θ)) * Vec2(-sin(θ), cos(θ))),
    # two lobes (size ~1 × 1.5) joined by a long neck of half-width 0.05
    "dumbbell" => (θ -> Vec2(2cos(θ), sin(θ) * (0.05 + 3cos(θ)^6)),
                   θ -> Vec2(-2sin(θ), cos(θ) * (0.05 + 3cos(θ)^6) - 18sin(θ)^2 * cos(θ)^5)),
]

function exact_area_perimeter(γ, dγ; M = 40_000)
    A = 0.0; P = 0.0
    for i in 0:M-1
        θ = 2π * i / M
        g = γ(θ); d = dγ(θ)
        A += 0.5 * (g[1] * d[2] - g[2] * d[1]); P += norm(d)
    end
    A * 2π / M, P * 2π / M
end

"Plain p×p Gauss–Legendre ∫ f J ds dt over every panel (NOT ad.w, which has the Jacobi weight)."
function mesh_integral(f, ad, p)
    xg, wg = gausslegendre(p)
    acc = 0.0
    for P in ad.panels, j in 1:p, i in 1:p
        x, _, _, J = panel_map(P, xg[i], xg[j])
        acc += wg[i] * wg[j] * f(x) * J
    end
    acc
end

"Number of panels containing x (Newton inversion of the ruled map, bbox prefilter)."
function coverage(ad, bbs, x)
    cnt = 0
    for (k, P) in enumerate(ad.panels)
        lo, hi = bbs[k]
        (lo[1] <= x[1] <= hi[1] && lo[2] <= x[2] <= hi[2]) || continue
        # Newton from two starts (edge panels around a hole are concave on the film side, and
        # a tight clamp can stall the iteration there)
        found = false
        for st0 in (Vec2(0.0, 0.0), Vec2(0.0, -0.9))
            st = st0; conv = false
            for _ in 1:60
                y, xs, xt, J = panel_map(P, st[1], st[2])
                δ = hcat(xs, xt) \ (y - x)
                st = st - δ
                st = Vec2(clamp(st[1], -4.0, 4.0), clamp(st[2], -4.0, 4.0))
                # also stop on a rounding-level residual: on the thin graded edge layers the
                # parameter step stalls above 1e-13 (|∂x/∂t| is small), which made random
                # points in those layers count as uncovered now and then
                if norm(δ) < 1e-13 || norm(y - x) < 1e-14 * (1 + norm(x))
                    conv = true; break
                end
            end
            if conv && maximum(abs, st) <= 1 + 1e-10
                found = true; break
            end
        end
        found && (cnt += 1)
    end
    cnt
end

function panel_bbox(P; n = 12)
    pts = [panel_map(P, s, t)[1] for s in range(-1, 1, length = n) for t in (-1.0, 1.0)]
    append!(pts, [panel_map(P, s, t)[1] for s in (-1.0, 1.0) for t in range(-1, 1, length = n)])
    lo = Vec2(minimum(q[1] for q in pts), minimum(q[2] for q in pts))
    hi = Vec2(maximum(q[1] for q in pts), maximum(q[2] for q in pts))
    pad = 0.05 * norm(hi - lo)
    (lo .- pad, hi .+ pad)
end

const p = 10

# the panel rules (src/geometry/panels.jl): Gauss–Legendre against the exact moments
# ∫ x^k dx = 2/(k+1) (k even), to degree 2n − 1
@testset "Gauss–Legendre rule: exact moments to degree 2n − 1" begin
    worst = 0.0
    for n in (1, 2, 3, 5, 10, 20, 40, 64)
        x, w = gausslegendre(n)
        @test all(w .> 0) && issorted(x) && x ≈ -reverse(x)
        worst = max(worst, maximum(abs(sum(w .* x .^ k) - (iseven(k) ? 2 / (k + 1) : 0.0)) for k in 0:2n-1))
    end
    @test worst < 1e-14
end

# the edge-layer rule (weight (1+x)^{-1/2}, built from Gauss–Legendre on 2p points) against the exact
# moments ∫ (1+x)^{-1/2} x^k dx = Σⱼ C(k,j) (−1)^{k−j} 2^{j+1/2}/(j+½) (t = 1+x), to degree 2p − 1
@testset "Gauss–Jacobi (0, −½) rule: exact moments to degree 2p − 1" begin
    worst = 0.0
    for p in (1, 2, 6, 10, 16, 24)
        r = gj_rule(p)
        @test all(r.w .> 0) && all(-1 .< r.x .< 1) && issorted(r.x)
        for k in 0:2p-1
            m = sum(binomial(big(k), j) * (-1)^(k - j) * sqrt(big(2.0)) * big(2.0)^j / (j + big(0.5)) for j in 0:k)
            worst = max(worst, Float64(abs(sum(r.w .* r.x .^ k) - m)))
        end
    end
    @test worst < 1e-14
end

@testset "boundary + mesh" begin
    for (name, (γ, dγ)) in DOMAINS
        @testset "$name" begin
            Aex, Pex = exact_area_perimeter(γ, dγ)
            panels = discretize_curve(γ, p; tol=1e-12)
            @test all(length(P.c) == p for P in panels)
            L = panel_length.(panels); N = length(panels)
            @test all(max(L[k], L[mod1(k + 1, N)]) <= 2.0001 * min(L[k], L[mod1(k + 1, N)]) for k in 1:N)
            @test abs(curve_area(panels) - Aex) < 1e-11 * max(1, Aex)

            bd, ad, info = build_mesh(panels, p)
            # boundary discretization
            @test abs(sum(bd.w) - Pex) < 1e-11 * Pex
            # edge quads: first N panels, in boundary order, on the boundary curve
            @test all(ad.panels[k].weighted && ad.panels[k].bpanel == k && ad.panels[k].B == panels[k].c for k in 1:N)
            nl = info.edge_layers
            # graded layers: panel N + (j−2)N + k sits over boundary panel k and continues the column
            @test all(!ad.panels[N+(j-2)*N+k].weighted && ad.panels[N+(j-2)*N+k].bpanel == k for j in 2:nl, k in 1:N)
            @test all(ad.panels[N+(j-2)*N+k].B == ad.panels[j == 2 ? k : N+(j-3)*N+k].T for j in 2:nl, k in 1:N)
            @test all(!P.weighted && P.bpanel == 0 for P in ad.panels[N*nl+1:end])
            @test length(ad.panels) == N * nl + info.ninterior
            # area: plain GL rule on J, against the exact area of γ
            Am = mesh_integral(x -> 1.0, ad, p)
            @test abs(Am - Aex) < 1e-10 * max(1, Aex)
            @test abs(Am - curve_area(panels)) < 1e-12 * max(1, Aex)
            # second moment ∫ x² dA = ∮ x³/3 dy
            Ix_ex = let M = 40_000
                sum(γ(2π * i / M)[1]^3 / 3 * dγ(2π * i / M)[2] for i in 0:M-1) * 2π / M
            end
            @test abs(mesh_integral(x -> x[1]^2, ad, p) - Ix_ex) < 1e-10 * max(1, abs(Ix_ex))
            # Jacobians positive everywhere (nodes and a fine grid), reasonable shapes
            q = mesh_quality(ad)
            @test all(ad.J .> 0)
            @test q.J_min > 0
            @test q.jratio_min > 0.05
            # interior < 12; graded edge layers are thinner by σ^{L−1} by design
            @test q.aspect_max_interior < 12
            @test q.aspect_max_edge < 12 / info.edge_grading^(info.edge_layers - 1)
            # panel centers inside Ω
            poly = [γ(θ) for θ in range(0, 2π, length = 20_001)[1:end-1]]
            @test all(point_in_polygon(panel_center(P), poly) for P in ad.panels)
            # coverage: random points covered exactly once inside, never outside
            bbs = [panel_bbox(P) for P in ad.panels]
            lo = Vec2(minimum(x[1] for x in poly), minimum(x[2] for x in poly))
            hi = Vec2(maximum(x[1] for x in poly), maximum(x[2] for x in poly))
            nin = 0; ok = true
            for i in 1:200
                x = lo + Vec2(rand(), rand()) .* (hi - lo)
                c = coverage(ad, bbs, x)
                inside = point_in_polygon(x, poly)
                inside && (nin += 1)
                ok &= c == (inside ? 1 : 0)
            end
            @test ok
            println(rpad(name, 10), " Nb=", N, "  edge=", q.nedge, "  interior=", q.ninterior,
                    "  |A−Aex|=", round(abs(Am - Aex), sigdigits = 2),
                    "  Jratio_min=", round(q.jratio_min, digits = 3),
                    "  aspect_max(edge/int)=", round(q.aspect_max_edge, digits = 2), "/", round(q.aspect_max_interior, digits = 2),
                    "  split tris=", info.nsplit,
                    "  Steiner on W=", info.nsteiner_W, "  repairs=", info.nfix,
                    "  t=", round(1e3 * info.times.total, digits = 1), " ms")
            if name == "dumbbell"
                # neck: |x| < 0.5 there; half-width ≈ 0.05
                neck = [k for k in 1:N if abs(curve_eval(panels[k], -1.0)[1][1]) < 0.5]
                @test !isempty(neck)
                @test all(0.03 < info.lfs[k] < 0.08 for k in neck)
                @test all(info.h[k] <= 0.35 * 0.06 for k in neck)
                @test all(L[k] <= 0.06 for k in neck)
                # strips from both sides of the neck do not meet: inner vertices stay on their side
                @test all(abs(info.W[k][2]) > 0.02 for k in neck)
            end
        end
    end

    @testset "domains with holes" begin
        circ(c, r) = θ -> c + r * Vec2(cos(θ), sin(θ))
        ell = θ -> Vec2(2cos(θ), 1.2sin(θ))
        CASES = [("annulus 2/1", circ(Vec2(0, 0), 2.0), [circ(Vec2(0, 0), 1.0)]),
                 ("eccentric", circ(Vec2(0, 0), 2.0), [circ(Vec2(0.6, 0.3), 0.8)]),
                 ("thin 1.2/1", circ(Vec2(0, 0), 1.2), [circ(Vec2(0, 0), 1.0)]),
                 ("two holes", ell, [circ(Vec2(-0.9, 0), 0.5), circ(Vec2(0.9, 0.1), 0.4)])]
        for (name, outer, hs) in CASES
            @testset "$name" begin
                op = discretize_curve(outer, p; tol=1e-12)
                hp = [reverse_curve(discretize_curve(h, p; tol=1e-12)) for h in hs]
                @test all(curve_area(h) < 0 for h in hp)
                Aex = curve_area(op) + sum(curve_area, hp)
                bd, ad, info = build_mesh(op, p; holes = hp)
                N = length(bd.panels)
                @test ncurves(bd) == 1 + length(hp) && ndrops(bd) == 1
                @test bd.cstart == cumsum([1; length(op); length.(hp)])
                @test all(ad.panels[k].weighted && ad.panels[k].bpanel == k && ad.panels[k].B == bd.panels[k].c for k in 1:N)
                # outward normals: away from the film, i.e. into each hole
                for (c, h) in enumerate(hs)
                    ctr = sum(h(2π * i / 64) for i in 0:63) / 64
                    @test all(dot(bd.n[i], ctr - bd.x[i]) > 0 for i in curve_nodes(bd, c + 1))
                end
                Am = mesh_integral(x -> 1.0, ad, p)
                @test abs(Am - Aex) < 1e-12 * Aex
                q = mesh_quality(ad)
                @test all(ad.J .> 0) && q.jratio_min > 0.05 && q.aspect_max_interior < 12
                # coverage: once inside the outer curve and outside every hole, never elsewhere
                polys = [[f(θ) for θ in range(0, 2π, length = 4001)[1:end-1]] for f in [outer; hs]]
                inside(x) = point_in_polygon(x, polys[1]) && !any(point_in_polygon(x, P) for P in polys[2:end])
                bbs = [panel_bbox(P) for P in ad.panels]
                lo = Vec2(minimum(x[1] for x in polys[1]), minimum(x[2] for x in polys[1]))
                hi = Vec2(maximum(x[1] for x in polys[1]), maximum(x[2] for x in polys[1]))
                ok = true
                for i in 1:200
                    x = lo + Vec2(rand(), rand()) .* (hi - lo)
                    ok &= coverage(ad, bbs, x) == (inside(x) ? 1 : 0)
                end
                @test ok
                println(rpad(name, 12), " Nb=", N, "  panels=", length(ad.panels),
                        "  |A−Aex|=", round(abs(Am - Aex), sigdigits = 2), "  repairs=", info.nfix)
            end
        end
        # discretize_curves: panels capped by the distance to the other curves (a narrow ring meshes),
        # separated curves left as by discretize_curve
        nr = discretize_curves([circ(Vec2(0, 0), 1.0), circ(Vec2(0, 0), 0.97)], p; tol=1e-12)
        @test all(panel_length(P) <= 4 * 0.03 + 1e-12 for c in nr for P in c)
        bd, ad, _ = build_mesh(nr[1], p; holes = [reverse_curve(nr[2])])
        @test abs(mesh_integral(x -> 1.0, ad, p) - π * (1 - 0.97^2)) < 1e-11
        wide = discretize_curves([circ(Vec2(0, 0), 2.0), circ(Vec2(0, 0), 1.0)], p; tol=1e-12)
        @test [P.c for c in wide for P in c] == [P.c for f in (circ(Vec2(0, 0), 2.0), circ(Vec2(0, 0), 1.0)) for P in discretize_curve(f, p; tol=1e-12)]
        # orientation and nesting are checked
        op = discretize_curve(θ -> 2Vec2(cos(θ), sin(θ)), p; tol=1e-12)
        @test_throws ErrorException build_mesh(op, p; holes = [discretize_curve(θ -> Vec2(cos(θ), sin(θ)), p; tol=1e-12)])
        @test_throws ErrorException build_mesh(op, p; holes = [reverse_curve(discretize_curve(θ -> Vec2(3 + cos(θ), sin(θ)), p; tol=1e-12))])
    end

    @testset "local feature size" begin
        panels = discretize_curve(θ -> Vec2(cos(θ), sin(θ)), p; lfs_factor = Inf, tol=1e-12)
        lf = local_feature_size(panels; radius_factor = 20)
        # unit circle: curvature bound 1, thickness term √2/2 (normals 90° apart)
        @test all(0.69 < l <= 1.0 for l in lf.vertex)
    end
end

# several drops (no solves): two near-circular drops at a few gaps
drop(c, ϕ) = θ -> c + (1 + 0.05cos(3(θ - ϕ))) * Vec2(cos(θ), sin(θ))
len(pans) = sum(panel_length(P) for P in pans)

@testset "multiple drops: mesh" begin
    @testset "input forms: one drop, a hole, two drops" begin
        # every form of `film_drops` gives the mesh of the single-drop call; one drop through the multi-drop path
        o = discretize_curve(θ -> 2Vec2(cos(θ), sin(θ)), p; tol=1e-12)
        h = reverse_curve(discretize_curve(θ -> Vec2(0.3, 0.1) + 0.8Vec2(cos(θ), sin(θ)), p; tol=1e-12))
        a = discretize_curve(drop(Vec2(-1.2, 0), 0.0), p; tol=1e-12); b = discretize_curve(drop(Vec2(1.2, 0), 0.0), p; tol=1e-12)
        same(m1, m2) = m1[1].x == m2[1].x && m1[1].pstart == m2[1].pstart && m1[1].cstart == m2[1].cstart &&
                       m1[2].x == m2[2].x && m1[2].w == m2[2].w && m1[2].pstart == m2[2].pstart &&
                       [P.bpanel for P in m1[2].panels] == [P.bpanel for P in m2[2].panels]
        @test film_drops(o) == [[o]] && film_drops(o; holes=[h]) == [[o, h]] && film_drops([a, b]) == [[a], [b]]
        @test film_drops([[o, h], [b]]) == [[o, h], [b]]
        @test_throws ErrorException film_drops([a, b]; holes=[h])
        m1 = build_mesh(film_drops(o), p; edge_layers=2)
        @test ndrops(m1[1]) == 1 && ndrops(m1[2]) == 1
        @test same(build_mesh(o, p; edge_layers=2), m1)
        @test same(build_mesh(o, p; holes=[h], edge_layers=2), build_mesh(film_drops(o; holes=[h]), p; edge_layers=2))
        @test same(build_mesh([a, b], p; edge_layers=2), build_mesh(film_drops([a, b]), p; edge_layers=2))
        # several drops, one with a hole
        d2 = [discretize_curve(θ -> Vec2(5 + cos(θ), sin(θ)), p; tol=1e-12)]
        bd, ad, _ = build_mesh([[o, h], d2], p)
        @test ndrops(bd) == 2 && ncurves(bd) == 3 && drop_curves(bd, 1) == 1:2 && drop_curves(bd, 2) == 3:3
        @test abs(mesh_integral(x -> 1.0, ad, p) - (curve_area(o) + curve_area(h) + curve_area(d2[1]))) < 1e-11
    end

    for g in (1.0, 0.25, 0.05)
        @testset "two drops, gap $g" begin
            c = 0.95 + g / 2   # facing points have r = 0.95
            curves = [discretize_curve(drop(Vec2(-c, 0), π), p; tol=1e-12), discretize_curve(drop(Vec2(c, 0), 0.0), p; tol=1e-12)]
            bd, ad, infos = build_mesh(curves, p; edge_layers=2)
            @test ndrops(bd) == 2 && ndrops(ad) == 2 && length(infos) == 2
            nd = node_drop(ad); pd = panel_drop(bd)
            for i in 1:2
                # nodes of drop i lie on its side of x = 0
                @test all(x -> sign(x[1]) == (i == 1 ? -1 : 1), ad.x[drop_nodes(ad, i)])
                @test all(x -> sign(x[1]) == (i == 1 ? -1 : 1), bd.x[drop_nodes(bd, i)])
                # drop i's mesh is exactly the mesh of its curve alone
                bdi, adi, _ = build_mesh(curves[i], p; edge_layers=2)
                @test ad.x[drop_nodes(ad, i)] == adi.x && ad.w[drop_nodes(ad, i)] == adi.w
                @test bd.x[drop_nodes(bd, i)] == bdi.x
                # area: a constant is not exact under the (1+t)^{-1/2} edge rule, hence the loose tolerance
                @test sum((ad.w ./ ad.wfac)[drop_nodes(ad, i)]) ≈ curve_area(curves[i]) rtol = 1e-4
                @test sum(bd.w[drop_nodes(bd, i)]) ≈ len(curves[i]) rtol = 1e-12
                # edge columns point at boundary panels of the same drop, each one exactly once
                # per layer (weighted layer: once)
                bp = [P.bpanel for P in ad.panels[drop_panels(ad, i)] if P.weighted]
                @test sort(bp) == collect(drop_panels(bd, i))
                @test all(pd[P.bpanel] == i for P in ad.panels[drop_panels(ad, i)] if P.bpanel > 0)
            end
            @test count(==(1), nd) == length(drop_nodes(ad, 1))
            # the quadrature-to-boundary map sees the right edge panels
            info = boundary_target_info(ad, bd)
            @test all(pd[b] == panel_drop(ad)[info[(b - 1) * p + 1][1]] for b in 1:npanels(bd))
        end
    end

    @testset "overlapping or nested curves are rejected" begin
        a = discretize_curve(drop(Vec2(0, 0), 0.0), p; tol=1e-12)
        @test_throws ErrorException build_mesh([a, discretize_curve(drop(Vec2(1.5, 0), 0.0), p; tol=1e-12)], p)
        @test_throws ErrorException build_mesh([a, discretize_curve(θ -> 0.3Vec2(cos(θ), sin(θ)), p; tol=1e-12)], p)
    end
end
