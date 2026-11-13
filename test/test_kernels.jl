# Kernels and fast sums: the 2D and 3D Stokeslets and their stress against closed forms and the Stokes
# equations (finite differences); the scalings of the FMM2D/FMM3D libraries; FMM, the 3D Ewald sum and the
# film Ewald sum (2D + 3D on one grid) against direct sums at tol 1e-6, without and with a remainder (the exact pair
# of common/symbol_pairs.jl). Kernels from Fourier symbols: test_symbol.jl. Timings and tighter tolerances are in
# validation/val_fmm.jl and val_ewald.jl.
#   julia -t 8 --project -e 'using Pkg; Pkg.test(test_args=["kernels"])'   (from src/OpenSurfaces)
include(joinpath(@__DIR__, "common", "symbol_pairs.jl"))
using OpenSurfaces, LinearAlgebra, StaticArrays, SparseArrays, Printf, Test, Random, QuadGK
using SpecialFunctions: besselj0, besselj
import FMM2D, FMM3D
using OpenSurfaces: _to2xN, _to3xN, _mhK01
using SpecialFunctions: besselk

relerr(a::Vector{Vec2}, b::Vector{Vec2}) = norm(reinterpret(Float64, a) - reinterpret(Float64, b)) / norm(reinterpret(Float64, b))
randpts(rng, n; L=1.0, c=Vec2(0, 0)) = [c + L * Vec2(rand(rng), rand(rng)) for _ in 1:n]
randvec(rng, n) = [Vec2(randn(rng), randn(rng)) for _ in 1:n]

@testset "kernels" begin
    rng = MersenneTwister(1)
    μ_2D = 1.7; μ_3D = 1.2
    for _ in 1:20
        r = Vec2(randn(rng), randn(rng)); ρ = norm(r)
        @test G2(r; μ_2D=μ_2D) ≈ (1 / (4π * μ_2D)) * (-log(ρ) * I + r * r' / ρ^2)
        @test G3(r; μ_3D=μ_3D) ≈ (1 / (4π * μ_3D)) * (I / ρ + r * r' / ρ^3)
        f = Vec2(randn(rng), randn(rng)); n = normalize(Vec2(randn(rng), randn(rng)))
        @test dot(P2(r), f) ≈ dot(r, f) / (2π * ρ^2)
        @test T2(r, n) * f ≈ stress2(r, f) * n
        @test issymmetric(Matrix(G2(r))) && issymmetric(Matrix(G3(r)))
    end
end

@testset "2D Stokeslet: stress from FD of G2, P2; Stokes equations" begin
    rng = MersenneTwister(2)
    μ_2D = 2.3; h = 1e-5
    for _ in 1:20
        x = Vec2(randn(rng), randn(rng)); f = Vec2(randn(rng), randn(rng))
        n = normalize(Vec2(randn(rng), randn(rng)))
        u(x) = G2(x; μ_2D=μ_2D) * f
        e1 = Vec2(h, 0); e2 = Vec2(0, h)
        J = hcat((u(x + e1) - u(x - e1)) / 2h, (u(x + e2) - u(x - e2)) / 2h)   # J[i,j] = ∂ⱼuᵢ
        p = dot(P2(x), f)
        σ = -p * I + μ_2D * (J + J')
        @test σ ≈ stress2(x, f) rtol = 1e-7
        @test norm(σ * n - T2(x, n) * f) < 1e-7 * norm(σ)
        @test abs(tr(J)) < 1e-8 * norm(J)                       # incompressible
        # −∇p + μ_2D Δu = 0 away from the source
        pf(y) = dot(P2(y), f)
        gp = Vec2((pf(x + e1) - pf(x - e1)) / 2h, (pf(x + e2) - pf(x - e2)) / 2h)
        hh = 1e-3; d1 = Vec2(hh, 0); d2 = Vec2(0, hh)
        lap = (u(x + d1) + u(x - d1) + u(x + d2) + u(x - d2) - 4u(x)) / hh^2
        @test norm(-gp + μ_2D * lap) < 1e-4 * norm(gp)
    end
    # total traction on a circle around the source equals −f (force balance; outward normal)
    f = Vec2(0.3, -1.1); R = 0.7; M = 64
    tot = sum(T2(R * Vec2(cos(θ), sin(θ)), Vec2(cos(θ), sin(θ))) * f * (2π * R / M) for θ in 2π * (0:M-1) / M)
    @test tot ≈ -f atol = 1e-12
end

@testset "library normalizations (single source)" begin
    rng = MersenneTwister(3)
    y = Vec2(0.1, -0.2); f = Vec2(0.7, -0.4)
    xs = [y + Vec2(randn(rng), randn(rng)) * s for s in (0.01, 1.0, 50.0)]
    v = FMM2D.stfmm2d(eps=1e-15, sources=_to2xN([y]), stoklet=_to2xN([f]), targets=_to2xN(xs), ppregt=3)
    for (k, x) in enumerate(xs)
        r = x - y; ρ = norm(r)
        @test Vec2(v.pottarg[:, k]) ≈ (r * r' / (2ρ^2) - log(ρ) / 2 * I) * f rtol = 1e-13
        @test v.pretarg[k] ≈ dot(r, f) / ρ^2 rtol = 1e-13
        # gradtarg[a,b] = ∂_a U_b : compare with analytic gradient of library kernel
        Ul(z) = (z * z' / (2 * dot(z, z)) - log(norm(z)) / 2 * I) * f
        hh = 1e-6 * ρ
        Jl = hcat((Ul(r + Vec2(hh, 0)) - Ul(r - Vec2(hh, 0))) / 2hh, (Ul(r + Vec2(0, hh)) - Ul(r - Vec2(0, hh))) / 2hh)
        @test v.gradtarg[:, :, k] ≈ Jl' rtol = 1e-6
    end
    w = FMM3D.stfmm3d(1e-15, _to3xN([y]); stoklet=_to3xN([f]), targets=_to3xN(xs), ppregt=1)
    for (k, x) in enumerate(xs)
        r = x - y; ρ = norm(r)
        @test Vec2(w.pottarg[1:2, k]) ≈ (I / ρ + r * r' / ρ^3) * f / (8π) rtol = 1e-13
        @test w.pottarg[3, k] == 0
    end
end

@testset "interleave" begin
    v = [Vec2(1, 2), Vec2(3, 4)]
    @test interleave(v) == [1.0, 2, 3, 4]
    @test deinterleave(interleave(v)) == v
end

@testset "FMM vs direct" begin
    rng = MersenneTwister(4)
    μ_2D = 1.3; μ_3D = 0.9
    # a few domain scales: the 2D log term would expose any missing log-scale constant
    for (N, M, L) in ((2000, 1500, 1.0), (3000, 0, 1e-3), (3000, 0, 1e3), (20000, 5000, 1.0))
        src = randpts(rng, N; L=L, c=Vec2(-0.3L, 0.2L)); f = randvec(rng, N)
        trg = M == 0 ? nothing : randpts(rng, M; L=1.2L, c=Vec2(-0.4L, 0.1L))
        nt = M == 0 ? N : M
        nrm = [normalize(Vec2(randn(rng), randn(rng))) for _ in 1:nt]
        a = stokes2d_apply(src, f, trg; μ_2D=μ_2D, eps=1e-12, traction_normals=nrm)
        d = stokes2d_direct(src, f, trg; μ_2D=μ_2D, traction_normals=nrm, eps=1e-12)
        eu = relerr(a.u, d.u); et = relerr(a.t, d.t)
        b = stokes3d_plane_apply(src, f, trg; μ_3D=μ_3D, eps=1e-12)
        e = stokes3d_plane_direct(src, f, trg; μ_3D=μ_3D, eps=1e-12)
        e3 = relerr(b.u, e.u)
        uzrel = b.uz_max / maximum(norm, e.u)
        println("N=$N M=$M L=$L: 2D u relerr=$(round(eu, sigdigits=3)) t relerr=$(round(et, sigdigits=3)); 3D relerr=$(round(e3, sigdigits=3)) uz/|u|=$(uzrel)")
        @test eu < 1e-10
        @test et < 1e-10
        @test e3 < 1e-10
        @test uzrel < 1e-12
        # no traction requested
        @test stokes2d_apply(src, f, trg; μ_2D=μ_2D, eps=1e-12).t === nothing
    end
end

@testset "self / coincident points" begin
    rng = MersenneTwister(5)
    N = 500
    src = randpts(rng, N); f = randvec(rng, N)
    # explicit targets equal to the sources: libraries drop coincident pairs
    a = stokes2d_apply(src, f, copy(src); eps=1e-12)
    b = stokes2d_apply(src, f; eps=1e-12)
    d = stokes2d_direct(src, f; eps=1e-12)
    @test relerr(a.u, d.u) < 1e-10 && relerr(b.u, d.u) < 1e-10
    a3 = stokes3d_plane_apply(src, f, copy(src); eps=1e-12); d3 = stokes3d_plane_direct(src, f; eps=1e-12)
    @test relerr(a3.u, d3.u) < 1e-10
    # target set containing some source points and some fresh points
    trg = vcat(src[1:10], randpts(rng, 10))
    @test relerr(stokes2d_apply(src, f, trg; eps=1e-12).u, stokes2d_direct(src, f, trg; eps=1e-12).u) < 1e-10
    @test relerr(stokes3d_plane_apply(src, f, trg; eps=1e-12).u, stokes3d_plane_direct(src, f, trg; eps=1e-12).u) < 1e-10
    # duplicated source points (two sources at the same location): both dropped at that target
    src2 = vcat(src, src[1:5]); f2 = vcat(f, randvec(rng, 5))
    @test relerr(stokes2d_apply(src2, f2; eps=1e-12).u, stokes2d_direct(src2, f2; eps=1e-12).u) < 1e-10
    @test relerr(stokes3d_plane_apply(src2, f2; eps=1e-12).u, stokes3d_plane_direct(src2, f2; eps=1e-12).u) < 1e-10
    # interleaved-density methods
    fi = interleave(f); nrm = [Vec2(1, 0) for _ in 1:N]
    r = stokes2d_apply(src, fi; traction_normals=nrm, eps=1e-12)
    @test r.u isa Vector{Float64} && length(r.u) == 2N
    @test relerr(deinterleave(r.u), d.u) < 1e-10
    @test relerr(deinterleave(r.t), stokes2d_direct(src, f; traction_normals=nrm, eps=1e-12).t) < 1e-10
    @test relerr(deinterleave(stokes3d_plane_apply(src, fi; eps=1e-12).u), d3.u) < 1e-10
    @test stokes2d_direct(src, fi; eps=1e-12).u ≈ interleave(d.u)
    # empty
    @test isempty(stokes2d_apply(Vec2[], Vec2[], Vec2[]; eps=1e-12).u)
end

@testset "Ewald: coincident targets and duplicated sources are skipped, tol 1e-6" begin
    # explicit targets on source nodes must not pick up the long range's K_L(0) f (≈ 1e-2 relative if they did)
    rng = MersenneTwister(15); tol = 1e-6; μ_3D = 1.5
    src = randpts(rng, 3000; L=2.0); f = randvec(rng, 3000) ./ 3000
    trg = vcat(src[1:400], randpts(rng, 200; L=2.0), src[2000:2100])
    e1 = relerr(ewald_apply(EwaldStokes3D(src, trg; μ_3D, tol), f), stokes3d_plane_direct(src, f, trg; μ_3D, eps=1e-12).u)
    e2 = relerr(ewald_apply(EwaldStokes3D(src, copy(src); μ_3D, tol), f), stokes3d_plane_direct(src, f; μ_3D, eps=1e-12).u)
    src2 = vcat(src, src[1:5]); f2 = vcat(f, randvec(rng, 5) ./ 3000)
    e3 = relerr(ewald_apply(EwaldStokes3D(src2; μ_3D, tol), f2), stokes3d_plane_direct(src2, f2; μ_3D, eps=1e-12).u)
    # the evaluator at explicit targets equal to the area nodes (its default fast sum is Ewald)
    p = 10
    bd, ad, _ = build_mesh(discretize_curve(θ -> Vec2(1.5cos(θ), 0.75sin(θ)), p; tol=1e-6), p)
    F = [Vec2(cos(x[2]), x[1]) for x in ad.x]
    E = Evaluator(Stokeslet3D(), ad; targets=ad.x, tol)
    e4 = relerr(E(F), Evaluator(Stokeslet3D(), ad; targets=ad.x, fast=:direct, tol)(F))
    e5 = relerr(E(F), Evaluator(Stokeslet3D(), ad; tol)(F))               # the same as targets = :area
    @printf("Ewald, coincident points: subset %.1e, targets = sources %.1e, duplicated %.1e; evaluator %.1e, against :area %.1e\n",
            e1, e2, e3, e4, e5)
    @test max(e1, e2, e3, e4, e5) < 5tol
    @test_throws ArgumentError FilmEwald(src, src[1:10], [Vec2(1, 0) for _ in 1:10])
end

relmax(u, v) = maximum(norm.(u .- v)) / maximum(norm.(v))
@testset "Ewald sums vs direct, without and with a remainder, tol 1e-6" begin
    # random points in a 3 × 1.5 box, and a mesh-like set clustered toward an ellipse's edge; the remainder is the
    # exact tensor pair of common/symbol_pairs.jl (at μ = 1, so it enters as R/μ_3D)
    rng = MersenneTwister(6); tol = 1e-6; μ_2D, μ_3D = 3.0, 2.0
    sets = (("uniform", [Vec2(3rand(rng) - 1.5, 1.5rand(rng) - 0.75) for _ in 1:4000]),
            ("clustered", [Vec2(1.5cos(t) * (1 - 0.3s^4), 0.75sin(t) * (1 - 0.3s^4)) for (t, s) in zip(2π .* rand(rng, 4000), rand(rng, 4000))]))
    trg = [Vec2(2rand(rng) - 1, rand(rng) - 0.5) for _ in 1:1000]
    # film Ewald: the clustered area points and boundary points on an ellipse around them
    θ = 2π .* (1:400) ./ 400; xb = [Vec2(1.6cos(t), 0.8sin(t)) for t in θ]; nb = [normalize(Vec2(cos(t) / 1.6, sin(t) / 0.8)) for t in θ]
    fb = [Vec2(randn(rng), randn(rng)) for _ in xb] ./ 400
    for (rname, R) in (("no remainder", nothing), ("remainder", tensorK()))
        for (name, src) in sets
            f = [Vec2(randn(rng), randn(rng)) for _ in src]
            uref = stokes3d_plane_direct(src, f; μ_3D, eps=1e-12).u
            tref = stokes3d_plane_direct(src, f, trg; μ_3D, eps=1e-12).u
            if R !== nothing
                uref = uref .+ rem_sum(src, f, src) ./ μ_3D; tref = tref .+ rem_sum(src, f, trg) ./ μ_3D
            end
            e = relmax(ewald_apply(EwaldStokes3D(src; μ_3D, remainder=R, tol), f), uref)
            et = relmax(ewald_apply(EwaldStokes3D(src, trg; μ_3D, remainder=R, tol), f), tref)
            @printf("Ewald 3D, %-12s %-9s: self %.1e, targets %.1e\n", rname, name, e, et)
            @test e < 20tol && et < 20tol
            name == "clustered" || continue
            # film Ewald: velocity at the area points from area and boundary sources, interior traction on Γ
            ufe = stokes2d_direct(src, f; μ_2D, eps=1e-12).u .+ stokes2d_direct(xb, fb, src; μ_2D, eps=1e-12).u .+ uref
            tfe = stokes2d_direct(src, f, xb; μ_2D, traction_normals=nb, eps=1e-12).t
            u, t = film_ewald_apply(FilmEwald(src, xb, nb; μ_2D, μ_3D, remainder=R, tol), f, fb)
            eu, ef = relmax(u, ufe), relmax(t, tfe)
            @printf("film Ewald, %-12s: velocity %.1e, traction %.1e\n", rname, eu, ef)
            @test eu < 20tol && ef < 20tol
        end
    end
end

@testset "K₀, zK₁ table of the modified Helmholtz kernel" begin
    zs = 10 .^ range(-10, log10(600), length=2001)
    @test maximum(abs(_mhK01(z)[1] / besselk(0, z) - 1) for z in zs) < 1e-13
    @test maximum(abs(_mhK01(z)[2] / (z * besselk(1, z)) - 1) for z in zs) < 1e-13
end
