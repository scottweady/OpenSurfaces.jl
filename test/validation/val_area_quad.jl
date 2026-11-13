# Validation (convergence) for src/quadrature/area_quad.jl, the sweeps behind the regression test
# test_quadrature.jl:
#   1. single unweighted panels (straight, curved, a graded-layer and an interior panel of a star mesh) vs
#      the exact integral by the divergence theorem (four scalar kernels, all target categories), p = 4…12
#   2. pure quadrature error (straight panel, exactly representable density, p = 10) at tol 1e-10, 1e-12
#   3. all self nodes of the straight and the graded-layer panel vs the exact integral (p = 8)
#   4. global disk tests at p = 8, 10, 12 (the weighted edge layer, against closed forms)
#   AREA_QUAD_TIMING=1: also the correction assembly time on 16² and 45² panel grids.
#   julia --project=. -t 8 src/OpenSurfaces/test/validation/val_area_quad.jl < /dev/null

include(joinpath(@__DIR__, "..", "common", "area_quad_common.jl"))
const VERBOSE = true

println("="^100)
println("Single panels vs the exact integral (rel. error, max over the 4 scalar kernels log ρ, 1/ρ, r₁r₂/ρ², r₁r₂/ρ³)")
println("="^100)

@testset "single panels vs the exact integral" begin
    for (name, P) in test_panels
        VERBOSE && @printf("%-14s %-24s %10s %10s %10s %10s %10s\n", name, "target", "p=4", "p=6", "p=8", "p=10", "p=12")
        flush(stdout)
        for (lab, kind, st, δ) in panel_targets(P)
            x0 = target_point(P, kind, st, δ)
            ref = exact_integral(P, x0, μphys)
            on = kind == :on ? (st[1], st[2]) : :auto
            errs = Float64[]
            for p in (4, 6, 8, 10, 12)
                ad = AreaDisc([P], p)
                μn = [poly_eval(μphys, x) for x in ad.x]
                W = area_local_weights(ad, 1, x0, Kscal; on=on, tol=1e-10)
                push!(errs, maximum(abs.(sum(packed(W[m]) * μn[m] for m in 1:p^2) - ref)) / maximum(abs.(ref)))
            end
            VERBOSE && @printf("%-14s %-24s %10.1e %10.1e %10.1e %10.1e %10.1e\n", "", lab, errs...)
            flush(stdout)
            @test errs[4] < 1e-9        # p = 10 (includes the interpolation error of μ̃ on curved panels)
            @test errs[5] < 1e-10       # p = 12
        end
    end
end

# straight panel: the bilinear map makes the degree-4 density exactly representable at p ≥ 5, so this is
# the pure quadrature error
@testset "pure quadrature error (straight panel, p = 10)" begin
    p = 10; P = test_panels[1][2]; worst = 0.0
    ad = AreaDisc([P], p)
    μn = [poly_eval(μphys, x) for x in ad.x]
    for (lab, kind, st, δ) in panel_targets(P)
        x0 = target_point(P, kind, st, δ)
        ref = exact_integral(P, x0, μphys)
        for tol in (1e-10, 1e-12)
            W = area_local_weights(ad, 1, x0, Kscal; on=kind == :on ? (st[1], st[2]) : :auto, tol=tol)
            err = maximum(abs.(sum(packed(W[m]) * μn[m] for m in 1:p^2) - ref)) / maximum(abs.(ref))
            worst = max(worst, err)
            @test err < 20tol + 1e-12
        end
    end
    VERBOSE && println("pure quadrature error, max over targets and tol ∈ {1e-10, 1e-12}: ", worst)
    flush(stdout)
end

# self targets at all nodes of the straight panel (exact density) and of a graded-layer panel
@testset "all self nodes vs the exact integral (p = 8)" begin
    p = 8
    for (name, P) in test_panels[[1, 3]]
        ad = AreaDisc([P], p)
        μn = [poly_eval(μphys, x) for x in ad.x]
        worst = 0.0
        for m in 1:p^2
            st = ad.st[m]
            W = area_local_weights(ad, 1, ad.x[m], Kscal; on=(st[1], st[2]), tol=1e-10)
            val = sum(packed(W[j]) * μn[j] for j in 1:p^2)
            ref = exact_integral(P, ad.x[m], μphys)
            worst = max(worst, maximum(abs.(val - ref)) / maximum(abs.(ref)))
        end
        VERBOSE && println("self nodes $name: max rel err = ", worst)
        flush(stdout)
        @test worst < (name == "straight" ? 1e-10 : 1e-8)
    end
end

println("="^100)
println("Disk: S3D[(2π√(1−r²))⁻¹ e_x] = 3/32 e_x ; S_Lap[4/(π√(1−r²))] = 1  (abs. errors)")
println("="^100)
@testset "disk" begin
    for (K, rin, p) in ((8, 0.78, 8), (8, 0.78, 10), (8, 0.78, 12))
        bp, ap = disk_mesh(K, rin)
        ad = AreaDisc(ap, p); bd = BoundaryDisc(bp, p)
        N = length(ad)
        μt = zeros(2N); σt = zeros(2N)
        for i in 1:N
            r2 = min(sum(abs2, ad.x[i]), 1.0)
            μt[2i-1] = 1 / (2π * sqrt(1 - r2)) / ad.wfac[i]       # μ̃ = μ (1+t)^{1/2} on edge quads
            σt[2i-1] = 4 / (π * sqrt(1 - r2)) / ad.wfac[i]
        end
        info = boundary_target_info(ad, bd)
        @test all(first.(info) .> 0)
        tA = @elapsed C = area_correction(ad, G3k; tol=1e-10)
        u = smooth_apply(ad, G3k, μt, ad.x; selfmode=true) + C * μt
        eA = max(maximum(abs.(u[1:2:end] .- 3 / 32)), maximum(abs.(u[2:2:end])))
        tB = @elapsed Cb = area_correction(ad, G3k; targets=bd.x, target_info=info, tol=1e-10)
        ub = smooth_apply(ad, G3k, μt, bd.x; selfmode=false) + Cb * μt
        eB = max(maximum(abs.(ub[1:2:end] .- 3 / 32)), maximum(abs.(ub[2:2:end])))
        CL = area_correction(ad, Lapk; tol=1e-10)
        uL = smooth_apply(ad, Lapk, σt, ad.x; selfmode=true) + CL * σt
        eL = maximum(abs.(uL[1:2:end] .- 1))
        CLb = area_correction(ad, Lapk; targets=bd.x, tol=1e-10)       # auto-detection of Γ targets
        uLb = smooth_apply(ad, Lapk, σt, bd.x; selfmode=false) + CLb * σt
        eLb = maximum(abs.(uLb[1:2:end] .- 1))
        VERBOSE && @printf("K=%d rin=%.2f p=%2d panels=%d N=%5d | G3: area %.1e  Γ %.1e | Laplace: area %.1e  Γ %.1e | t_corr %.2fs (Γ %.2fs)\n",
                           K, rin, p, npanels(ad), N, eA, eB, eL, eLb, tA, tB)
        flush(stdout)
        if p == 10
            @test eA < 5e-10
            @test eB < 5e-10
            @test eL < 5e-9
            @test eLb < 5e-9
        elseif p == 12
            @test eA < 5e-11 && eB < 5e-11
        end
    end
end

if get(ENV, "AREA_QUAD_TIMING", "0") == "1"
    for n in (16, 45)
        ad = AreaDisc(grid_mesh(n), 10)
        st = Dict()
        t = @elapsed area_correction(ad, G3k; stats=st, transposed=true, tol=1e-10)
        @printf("timing: %d panels, p=10, %d threads: %.2f s, %.2f near panels/target, %d kernel evals/pair, nnz %.2e\n",
                npanels(ad), Threads.nthreads(), t, st[:pairs] / length(ad), round(Int, st[:kernel_evals] / st[:pairs]), st[:nnz])
    end
end
