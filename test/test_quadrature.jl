# Quadrature at the production p = 10: the boundary single layer and its traction (src/quadrature/
# line_quad.jl) against QuadGK and exact identities; the area potentials (src/quadrature/area_quad.jl)
# on single panels against exact integrals (divergence theorem) and on the disc against closed forms. Convergence in
# p and K: validation/val_line_quad.jl, val_area_quad.jl.
#   julia -t 8 --project -e 'using Pkg; Pkg.test(test_args=["quadrature"])'   (from src/OpenSurfaces)
include(joinpath(@__DIR__, "common", "line_quad_common.jl"))
include(joinpath(@__DIR__, "common", "area_quad_common.jl"))
using SpecialFunctions: gamma, ellipe

@testset "line_quad" begin

@testset "log moments" begin
    for t in (-1.0, -0.97, -0.3, 0.0, 0.5, 0.99, 1.0)          # ±1: a target at a panel end
        M = _lq_log_moments(t, 30)
        pts = abs(t) == 1 ? (-1, 1) : (-1, t, 1)
        err = maximum(abs(M[k+1] - quadgk(s -> log(abs(t - s)) * legendre_P(k, s), pts...;
                                         rtol=1e-15, atol=1e-16)[1]) for k in 0:29)
        @test err < 1e-13
    end
end

@testset "circle: S[n] = 0 inside and on Γ, traction = n (p = −1)" begin
    for p in (10,)
        bd = curve_disc(circle(), 16, p); lq = LineQuad_(bd; tol=1e-10)
        ν = copy(bd.n)
        # on Γ
        A = slp2d_velocity_matrix(lq)
        @test maximum(abs, A * flat(ν)) < 1e-13
        # inside, incl. 1e-8 × panel length from Γ, through smooth + sparse correction
        tg = near_targets(bd, lq, (1, 5), (-1.0, -0.6, 0.0, 0.97), (0.3, 1e-2, 1e-5, 1e-8))
        push!(tg, Vec2(0, 0), Vec2(0.3, -0.2))
        S = line_smooth_matrix(r -> G2t(r), bd, tg)
        C = slp2d_correction(lq, tg)
        u = vecs(S * flat(ν) + C * flat(ν))
        @test maximum(norm, u) < 1e-13
        @test maximum(norm, vecs(S * flat(ν))) > 1e-3        # the smooth rule alone is wrong
        # pressure p = (1/2π)∮ (r·n)/ρ² ds = −1 inside
        # (a 1/ρ kernel evaluated directly at distance d loses ~eps/d digits to rounding,
        #  so the closest targets are excluded here)
        tgp = near_targets(bd, lq, (1, 5), (-1.0, -0.6, 0.0, 0.97), (0.3, 1e-2, 1e-5))
        pr = [line_potential_eval(P2row, lq, x, ν)[1] for x in tgp]
        @test maximum(abs, pr .+ 1) < 1e-10
        # traction −p n = n on Γ
        T = traction_self_matrix(bd)
        @test maximum(norm, vecs(T * flat(ν)) .- bd.n) < 1e-13
    end
end

@testset "S[n] = 0, traction = n on a star" begin
    bd = curve_disc(star(), 64, 12); lq = LineQuad_(bd; tol=1e-10)
    ν = copy(bd.n)
    @test maximum(abs, slp2d_velocity_matrix(lq) * flat(ν)) < 1e-11
    @test maximum(norm, vecs(traction_self_matrix(bd) * flat(ν)) .- bd.n) < 1e-10
    tg = near_targets(bd, lq, (1, 9, 40), (-0.99, 0.2), (1e-2, 1e-8))
    @test maximum(norm, slp2d_eval(lq, tg, ν)) < 1e-11
end

@testset "constant density on a circle of radius a" begin
    # u inside = (1/(4πη)) (−2πa log a + πa) e_x = (a/η)(1/4 − ½ log a) e_x
    for a in (0.5, 2.0), μ_2D in (1.0, 2.5)
        bd = curve_disc(circle(a), 16, 10); lq = LineQuad_(bd; tol=1e-10)
        ν = fill(Vec2(1, 0), length(bd))
        uex = Vec2(a / μ_2D * (0.25 - 0.5 * log(a)), 0)
        uΓ = vecs(slp2d_velocity_matrix(lq; μ_2D=μ_2D) * flat(ν))
        @test maximum(norm, uΓ .- Ref(uex)) < 1e-13 * max(1, a)
        tg = near_targets(bd, lq, (2, 11), (-1.0, 0.0, 0.5), (0.5, 1e-3, 1e-8))
        push!(tg, Vec2(0, 0))
        u = vecs(line_smooth_matrix(r -> G2t(r, μ_2D), bd, tg) * flat(ν) +
                 slp2d_correction(lq, tg; μ_2D=μ_2D) * flat(ν))
        @test maximum(norm, u .- Ref(uex)) < 1e-13 * max(1, a)
    end
end

@testset "smooth curves vs QuadGK (on Γ and near)" begin
    for (f, K, p, tol) in ((ellipse(), 32, 10, 1e-12), (star(), 64, 10, 1e-12))
        bd = curve_disc(f, K, p); lq = LineQuad_(bd; tol=1e-10); ν = smooth_density(bd)
        A = slp2d_velocity_matrix(lq)
        u = vecs(A * flat(ν))
        e_on = maximum(norm(u[it] - ref_total_node(G2t, bd, it, ν)) for it in (1, 2, p ÷ 2, p, p + 1, 3p + 4))
        tg = near_targets(bd, lq, (1, 7), (-1.0, -0.999, 0.0, 0.31, 0.9999), (1e-1, 1e-2, 1e-4, 1e-8))
        append!(tg, near_targets(bd, lq, (3,), (-0.5, 0.5), (1e-2, 1e-6); side=+1))   # exterior
        un = vecs(line_smooth_matrix(r -> G2t(r), bd, tg) * flat(ν) + slp2d_correction(lq, tg) * flat(ν))
        e_near = maximum(norm(un[m] - ref_total(G2t, bd, tg[m], ν)) for m in eachindex(tg))
        @test e_on < tol
        @test e_near < tol
    end
end

@testset "traction: diagonal limit, PV accuracy, jump +½, interior limit" begin
    # diagonal limit lim_{y→x} T2(x−y, n(x)) = −(κ/2π) τ τᵀ  (error O(|y−x|))
    bd = curve_disc(star(), 64, 12)
    for it in (3, 50, 200)
        k = (it - 1) ÷ 12 + 1; si = bd.rule.x[it-(k-1)*12]
        lim = -(bd.κ[it] / (2π)) * (bd.τ[it] * bd.τ[it]')
        e = [norm(T2t(bd.x[it] - curve_eval(bd.panels[k], si + h)[1], bd.n[it]) - lim) for h in (1e-2, 1e-3)]
        @test e[2] < 0.15 * e[1]
        @test e[2] < 1e-3
    end
    for (f, K, p, tol) in ((circle(), 32, 10, 1e-11), (ellipse(), 32, 10, 1e-11), (star(), 64, 12, 1e-10))
        bd = curve_disc(f, K, p); lq = LineQuad_(bd; near_factor=2.0, tol=1e-10); ν = smooth_density(bd)
        t = vecs(traction_self_matrix(bd) * flat(ν))
        for it in (1, 5, p, 3p + 2)
            k = (it - 1) ÷ p + 1
            # vs ½ν + PV reference
            @test norm(t[it] - (0.5ν[it] + ref_traction_pv(bd, it, ν))) < tol
            # vs the limit from inside/outside (Richardson in the distance ε)
            ε = 1e-5 * lq.len[k]; n = bd.n[it]
            tin(e) = slp2d_traction_eval(lq, bd.x[it] - e * n, n, ν)
            tex(e) = slp2d_traction_eval(lq, bd.x[it] + e * n, n, ν)
            ti = 2tin(ε) - tin(2ε); te = 2tex(ε) - tex(2ε)
            @test norm(ti - t[it]) < 1e-8
            @test norm(ti - te - ν[it]) < 1e-8                                   # t_int − t_ext = +ν
            @test norm(te - vecs(traction_self_matrix(bd; jump=-0.5) * flat(ν))[it]) < 1e-8
        end
    end
end

@testset "force balance: ∮ t_int ds = 0, ∮ t_ext ds = −L f for constant f" begin
    for (f, K, p) in ((ellipse(), 32, 10), (star(), 64, 12))
        bd = curve_disc(f, K, p)
        fc = Vec2(0.3, -1.1); ν = fill(fc, length(bd))
        ti = vecs(traction_self_matrix(bd) * flat(ν))
        te = vecs(traction_self_matrix(bd; jump=-0.5) * flat(ν))
        L = sum(bd.w)
        @test norm(sum(bd.w .* ti)) < 1e-11
        @test norm(sum(bd.w .* te) + L * fc) < 1e-11
    end
end

@testset "near-region criterion keeps the smooth rule accurate" begin
    for p in (10,), (f, K) in ((ellipse(), 48), (star(), 48))
        bd = curve_disc(f, K, p); lq = LineQuad_(bd; tol=1e-10); ν = smooth_density(bd)
        W = Vector{Mat2}(undef, p); emax = 0.0
        for k in 1:4:npanels(bd), φ in range(0, 2π, length=37)
            x = lq.center[k] + 1.0001 * lq.rnear[k] * Vec2(cos(φ), sin(φ))
            idx = panel_nodes(bd, k)
            line_panel_weights!(W, r -> G2t(r), x, bd.panels[k], lq)
            emax = max(emax, norm(sum(W[j] * ν[idx[j]] for j in 1:p) -
                                  sum(G2t(x - bd.x[jn]) * bd.w[jn] * ν[jn] for jn in idx)))
        end
        @test emax < 1e-11
    end
end

@testset "correction layout and self exclusion" begin
    bd = curve_disc(ellipse(), 16, 10); lq = LineQuad_(bd; tol=1e-10)
    C = slp2d_self_correction(lq)
    S = line_smooth_matrix(r -> G2t(r), bd, bd.x; self=true)
    A = slp2d_velocity_matrix(lq)
    @test norm(A - (S + Matrix(C))) < 1e-14 * norm(A)
    @test all(S[2i-1:2i, 2i-1:2i] == zeros(2, 2) for i in 1:length(bd))
    @test size(C) == (2length(bd), 2length(bd))
    # symmetry of the Stokeslet blocks within each 2×2 block of the accurate matrix
    @test norm(A[1:2, 3:4] - A[1:2, 3:4]') < 1e-14
end

end

# area potentials
const VERBOSE = false

# local quadrature on single panels against the exact integral (divergence theorem), all target kinds:
# on the panel, near and on its sides and corners, off it down to 1e-8 of its size
@testset "single panels against the exact integral, p = 10" begin
    p = 10; worst = 0.0
    for (name, P) in test_panels
        ad = AreaDisc([P], p)
        μn = [poly_eval(μphys, x) for x in ad.x]
        for (lab, kind, st, δ) in panel_targets(P)
            x0 = target_point(P, kind, st, δ)
            on = kind == :on ? (st[1], st[2]) : :auto
            W = area_local_weights(ad, 1, x0, Kscal; on=on, tol=1e-10)
            val = sum(packed(W[m]) * μn[m] for m in 1:p^2)
            ref = exact_integral(P, x0, μphys)
            err = maximum(abs.(val - ref)) / maximum(abs.(ref))
            worst = max(worst, err)
            VERBOSE && @printf("%-14s %-24s %.1e\n", name, lab, err)
            @test err < 5e-10                     # max 3.7e-11
        end
    end
    @printf("single panels (straight, curved, graded layer, interior), all targets: max rel. error %.1e\n", worst)
end

@testset "disk, p = 10" begin
    for (K, rin, p) in ((8, 0.78, 10),)
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

@testset "assembly consistency" begin
    p = 8
    bp, ap = disk_mesh(6, 0.7)
    ad = AreaDisc(ap, p); bd = BoundaryDisc(bp, p)
    N = length(ad)
    μt = randn(2N)
    # target-dependent kernel: traction with the normal of target i (Γ nodes)
    Kt = (i, r) -> T2(r, bd.n[i])
    info = boundary_target_info(ad, bd)
    C1 = area_correction(ad, Kt; targets=bd.x, target_info=info, tol=1e-10)
    C2 = area_correction(ad, Kt; targets=bd.x, tol=1e-10)                      # detection only
    @test norm(C1 - C2, Inf) < 1e-13 * norm(C1, Inf)
    # rows of one Γ target vs direct local weights minus smooth rule
    i = 3
    k = info[i][1]
    W = area_local_weights(ad, k, bd.x[i], r -> T2(r, bd.n[i]); on=(info[i][2][1], -1.0), tol=1e-10)
    for (m, j) in enumerate(panel_nodes(ad, k))
        blk = W[m] - T2(bd.x[i] - ad.x[j], bd.n[i]) * ad.w[j]
        @test Matrix(C1[2i-1:2i, 2j-1:2j]) ≈ Matrix(blk) atol = 1e-14 * norm(blk) + 1e-15
    end
    # self rows: local weights with the smooth self term excluded
    C = area_correction(ad, G3k; tol=1e-10)
    i = 2 * p^2 + 17; k = (i - 1) ÷ p^2 + 1
    W = area_local_weights(ad, k, ad.x[i], G3k; on=(ad.st[i][1], ad.st[i][2]), tol=1e-10)
    for (m, j) in enumerate(panel_nodes(ad, k))
        blk = j == i ? W[m] : W[m] - G3k(ad.x[i] - ad.x[j]) * ad.w[j]
        @test Matrix(C[2i-1:2i, 2j-1:2j]) ≈ Matrix(blk) rtol = 1e-13
    end
    # transposed assembly
    Ct = area_correction(ad, G3k; transposed=true, tol=1e-10)
    @test Ct == sparse(transpose(C))
end

@testset "near zone floor (loose tol, high p) and Evaluator's area_correction options" begin
    # near_factor fell below 1 at loose tol or high p, so targets at a neighbor's corner got the smooth rule:
    # S[1/ω] = π/4 on the unit disc (stored density) had error 8.1e-3 at p = 14, tol 1e-4
    @test AreaQuadOpts(10; tol=1e-6).near_factor ≈ 1.23534 atol = 1e-5       # p = 10, tol 1e-6: unchanged
    @test AreaQuadOpts(14; tol=1e-4).near_factor == 1.2
    p = 14; tol = 1e-4
    _, ad, _ = build_mesh(discretize_curve(θ -> Vec2(cos(θ), sin(θ)), p; tol=1e-6), p)
    f = (1 ./ sqrt.(1 .- norm.(ad.x) .^ 2)) ./ ad.wfac
    e = maximum(abs.(Evaluator(LaplaceKernel3D(), ad; tol)(f) .- π / 4))
    # the options reach area_correction: a wider near zone, and the far sum on an upsampled rule
    p = 10; tol = 1e-6
    _, ad, _ = build_mesh(discretize_curve(θ -> Vec2(cos(θ), sin(θ)), p; tol=1e-6), p)
    f = (1 ./ sqrt.(1 .- norm.(ad.x) .^ 2)) ./ ad.wfac
    u0 = Evaluator(LaplaceKernel3D(), ad; tol)(f)
    E1 = Evaluator(LaplaceKernel3D(), ad; tol, near_factor=2.0)
    E2 = Evaluator(LaplaceKernel3D(), ad; tol, upsampling=AreaUpsampling(ad, 14))
    e1 = maximum(abs.(E1(f) .- π / 4)); e2 = maximum(abs.(E2(f) .- π / 4)); e0 = maximum(abs.(u0 .- π / 4))
    @printf("S[1/ω] = π/4: p = 14, tol 1e-4: %.1e; p = 10, tol 1e-6: default %.1e, near_factor 2 %.1e, upsampled (q = 14) %.1e\n",
            e, e0, e1, e2)
    @test e < tol && max(e0, e1, e2) < 10tol
    @test nnz(E1.C[1]) > nnz(Evaluator(LaplaceKernel3D(), ad; tol).C[1])
    @test_throws ArgumentError Evaluator(LaplaceKernel3D(), ad; density=:smooth, upsampling=AreaUpsampling(ad, 12))
end
