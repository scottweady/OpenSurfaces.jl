# FMM2D / FMM3D timings at N = 1e5 sources = targets (printed).
#   julia --project=. -t 8 src/OpenSurfaces/test/validation/val_fmm.jl
using OpenSurfaces, LinearAlgebra, Test, Random
randpts(rng, n; L=1.0, c=Vec2(0, 0)) = [c + L * Vec2(rand(rng), rand(rng)) for _ in 1:n]
randvec(rng, n) = [Vec2(randn(rng), randn(rng)) for _ in 1:n]

@testset "timings N=1e5" begin
    rng = MersenneTwister(6)
    N = 100_000
    src = randpts(rng, N); f = randvec(rng, N); nrm = [normalize(Vec2(randn(rng), randn(rng))) for _ in 1:N]
    stokes2d_apply(src[1:100], f[1:100]; eps=1e-12); stokes3d_plane_apply(src[1:100], f[1:100]; eps=1e-12)
    t1 = @elapsed stokes2d_apply(src, f; eps=1e-12)
    t2 = @elapsed stokes2d_apply(src, f; eps=1e-12, traction_normals=nrm)
    t3 = @elapsed stokes3d_plane_apply(src, f; eps=1e-12)
    t4 = @elapsed stokes2d_apply(src, f; eps=1e-8)
    t5 = @elapsed stokes3d_plane_apply(src, f; eps=1e-8)
    println("N=1e5 sources=targets: FMM2D u: $(round(t1, digits=3)) s, u+traction: $(round(t2, digits=3)) s, FMM3D: $(round(t3, digits=3)) s (eps=1e-12)")
    println("                        FMM2D u: $(round(t4, digits=3)) s, FMM3D: $(round(t5, digits=3)) s (eps=1e-8)")
    @test true
end
