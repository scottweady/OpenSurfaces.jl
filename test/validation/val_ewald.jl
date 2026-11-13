# Ewald/NUFFT in-plane 3D Stokeslet on mesh nodes: set-up and apply time against FMM3D, and the error
# against FMM3D at 1e-13 (printed).
#   run by run_all.jl (from the package root; `push!(ARGS, "ewald")` before its include runs this file only)
using OpenSurfaces, LinearAlgebra, Printf
relerr(u, v) = maximum(norm.(u .- v)) / maximum(norm.(v))
# timing vs FMM3D on mesh nodes
for maxlen in (Inf, 0.15)
    pans = discretize_curve(θ -> Vec2(1.5cos(θ), 0.75sin(θ)), 10; maxlen=maxlen, tol=1e-12)
    bd, ad, info = build_mesh(pans, 10)
    f = [Vec2(randn(), randn()) for _ in ad.x] .* ad.w
    for tol in (1e-6, 1e-10)
        tp = @elapsed E = EwaldStokes3D(ad.x; tol=tol)
        ewald_apply(E, f)
        te = minimum(@elapsed(ewald_apply(E, f)) for _ in 1:3)
        stokes3d_plane_apply(ad.x, f; eps=tol)
        tf = minimum(@elapsed(stokes3d_plane_apply(ad.x, f; eps=tol)) for _ in 1:3)
        e = relerr(ewald_apply(E, f), stokes3d_plane_apply(ad.x, f; eps=1e-13).u)
        @printf("N=%6d tol %.0e: Ewald setup %.2fs (%s), apply %.3fs | FMM3D %.3fs | speedup %.1f× | err vs FMM3D(1e-13) %.1e\n",
                length(ad), tol, tp, string(round.(values(E.timings) |> collect; digits=2)), te, tf, tf / te, e)
    end
end
