# The common evaluator (src/evaluators/evaluator.jl): for every kernel family the fast sum (Ewald, FMM3D,
# FMM2D, Helmholtz FMM) against the direct sum with the same corrections, area and line sources, stored and smooth
# densities; several target sets against one at a time; `evaluator_matrix` against the apply; the errors. The
# accuracy of the potentials themselves: test_volume.jl (closed forms on the disc).
#   julia -t 8 --project -e 'using Pkg; Pkg.test(test_args=["evaluator"])'   (from the package root)
include(joinpath(@__DIR__, "common", "symbol_pairs.jl"))
using OpenSurfaces, LinearAlgebra, Printf, Random, Test, QuadGK
using SpecialFunctions: besselk

relmax(u, v) = maximum(norm.(u .- v)) / maximum(norm.(v))
direct(K, src; kw...) = Evaluator(K, src; fast=:direct, kw...)

@testset "fast sums against direct sums, tol 1e-6" begin
    rng = MersenneTwister(4); tol = 1e-6; p = 10
    # the curve at the default tol (11.8k area nodes; 28.5k at 1e-10): this checks the evaluator's paths against the
    # same sum done directly, and the accuracy of the fast sums themselves is test_kernels.jl's
    pans = discretize_curve(θ -> (1 + 0.2cos(3θ)) * Vec2(cos(θ), sin(θ)), p; tol=1e-6)
    bd, ad, _ = build_mesh(pans, p; edge_layers=2)
    g = [sin(x[1]) + x[2]^2 for x in ad.x]; F = [Vec2(cos(x[2]), x[1]) for x in ad.x]; σ = [cos(2atan(x[2], x[1])) for x in bd.x]
    pts = [Vec2(0.9cos(t), 0.9sin(t)) for t in range(0, 2π; length=50)]
    e = Dict{String,Float64}()
    for (name, fast) in (("Stokeslet3D Ewald", :ewald), ("Stokeslet3D FMM", :fmm))
        E = Evaluator(Stokeslet3D(; μ_3D=2.0), ad; fast, tol, eps=tol); D = direct(Stokeslet3D(; μ_3D=2.0), ad; tol)
        e[name] = relmax(E(F), D(F))
    end
    Eb = Evaluator(Stokeslet3D(), ad; targets=:boundary, bd, tol, eps=tol)
    e["Stokeslet3D → Γ"] = relmax(Eb(F), direct(Stokeslet3D(), ad; targets=:boundary, bd, tol)(F))
    for (name, K) in (("LaplaceKernel3D", LaplaceKernel3D()), ("LaplaceKernel2D", LaplaceKernel2D()),
                      ("LaplaceGradient2D", LaplaceGradient2D()), ("BiharmonicGradient2D", BiharmonicGradient2D(; κ=0.3, R=1.1)),
                      ("MHKernel", MHKernel(3.0)))
        E = Evaluator(K, ad; density=:smooth, targets=(:area, :boundary), bd, tol, eps=tol)
        D = direct(K, ad; density=:smooth, targets=(:area, :boundary), bd, tol)
        (a, b), (da, db) = E(g), D(g)
        e[name] = max(relmax(a, da), relmax(b, db))
        # one fast sum for both sets gives the same as one set at a time
        ea = Evaluator(K, ad; density=:smooth, targets=:area, tol, eps=tol)(g)
        e[name*" (sets)"] = relmax(ea, a)
    end
    N = MHNormalKernel(3.0, bd.n)
    e["MHNormalKernel"] = relmax(Evaluator(N, ad; density=:smooth, targets=:boundary, bd, tol, eps=tol)(g),
                                 direct(N, ad; density=:smooth, targets=:boundary, bd, tol)(g))
    e["MHKernel, line sources"] = relmax(Evaluator(MHKernel(3.0), bd; targets=ad.x, tol, eps=tol)(σ),
                                         direct(MHKernel(3.0), bd; targets=ad.x, tol)(σ))
    e["points"] = relmax(Evaluator(LaplaceKernel3D(), ad; targets=pts, tol, eps=tol)(g),
                         direct(LaplaceKernel3D(), ad; targets=pts, tol)(g))
    for (k, v) in sort(collect(e)); @printf("evaluator %-28s fast − direct %.1e\n", k, v); end
    @test all(v < 10tol for v in values(e))
end

@testset "Stokeslet3D with smooth Vec2 densities: Ewald, FMM, direct, polar rays" begin
    # density = :smooth with a Vec2 force threw a MethodError at apply; it sums on the unweighted copy of the mesh,
    # whose interior nodes are the film nodes (coincident targets: the Ewald fix of test_kernels.jl)
    tol = 1e-6; p = 10; μ = 1.3; a, b = 1.2, 0.8
    bd, ad, _ = build_mesh(discretize_curve(θ -> Vec2(a * cos(θ), b * sin(θ)), p; tol=1e-6), p)
    F = [Vec2(cos(x[2]), x[1]^2) for x in ad.x]
    us = [Evaluator(Stokeslet3D(; μ_3D=μ), ad; density=:smooth, fast, tol)(F) for fast in (:ewald, :fmm, :direct)]
    e_fast = max(relmax(us[1], us[3]), relmax(us[2], us[3]))
    # uniform force (1, 0): u(x) = (1/4πμ) ∫ ρmax(φ) (I + e eᵀ) f dφ, trapezoid in φ (smooth for x inside)
    function ray(x; M=2048)
        acc = zero(Vec2)
        for j in 0:M-1
            φ = 2π * j / M; e = Vec2(cos(φ), sin(φ))
            A = (e[1] / a)^2 + (e[2] / b)^2; B = x[1] * e[1] / a^2 + x[2] * e[2] / b^2; C = (x[1] / a)^2 + (x[2] / b)^2 - 1
            ρm = (-B + sqrt(B^2 - A * C)) / A
            acc += ρm * (Vec2(1, 0) + e * e[1])
        end
        acc * (2π / M) / (4π * μ)
    end
    idx = [i for i in eachindex(ad.x) if (ad.x[i][1] / a)^2 + (ad.x[i][2] / b)^2 < 0.8^2][1:20:end]
    u1 = Evaluator(Stokeslet3D(; μ_3D=μ), ad; density=:smooth, tol)(fill(Vec2(1, 0), length(ad.x)))
    e_ray = relmax(u1[idx], ray.(ad.x[idx]))
    @printf("Stokeslet3D, smooth Vec2 density: fast − direct %.1e, uniform force against polar rays %.1e (%d targets)\n",
            e_fast, e_ray, length(idx))
    @test e_fast < 5tol && e_ray < 5tol
end

struct MyStokeslet end                                  # a tensor kernel of the user's, with its shape declared
(::MyStokeslet)(r::Vec2) = G3(r)
OpenSurfaces.kernel_shape(::MyStokeslet) = (2, 2)

@testset "kernel shapes: SymbolKernels, declared shapes, scalar returns, the probe" begin
    # a tensor SymbolKernel as the kernel gives the whole tensor (not its xx component only), a scalar one no DimensionMismatch
    tol = 1e-6; p = 10
    bd, ad, _ = build_mesh(discretize_curve(θ -> Vec2(cos(θ), 0.7sin(θ)), p; tol=1e-4), p)
    F = [Vec2(cos(x[2]), x[1]) for x in ad.x]; g = [1 + x[1] * x[2] for x in ad.x]
    T = tensorK()                                      # the exact pair of common/symbol_pairs.jl, base Stokeslet3D()
    uT = Evaluator(Stokeslet3D(), ad; remainder=T, tol)(F)
    e_tensor = relmax(Evaluator(T, ad; fast=:direct, tol)(F), uT)
    e_direct_rem = relmax(Evaluator(Stokeslet3D(), ad; remainder=T, fast=:direct, tol)(F), uT)   # keeps T
    Hd = 0.3; dep(k) = -1 / (k * (exp(2k * Hd) + 1))   # tanh(kH)/(2k) on LaplaceKernel3D
    S = SymbolKernel(k -> tanh(k * Hd) / (2k); base=LaplaceKernel3D(), scale=Hd, remainder=dep, outputs=(:value,))
    e_scalar = relmax(Evaluator(S, ad; density=:smooth, fast=:direct, tol)(g), single_layer(SingleLayer3D(ad; tol, remainder=S), g))
    e_declared = relmax(Evaluator(MyStokeslet(), ad; tol)(F), Evaluator(Stokeslet3D(), ad; fast=:direct, tol)(F))
    e_closure = relmax(Evaluator(r -> 1 / (4π * norm(r)), ad; tol)(g), Evaluator(LaplaceKernel3D(), ad; fast=:direct, tol)(g))
    @printf("kernel shapes: tensor SymbolKernel (direct) − Stokeslet3D + remainder (Ewald) %.1e, scalar SymbolKernel − SingleLayer3D %.1e, declared tensor %.1e, scalar closure %.1e\n", e_tensor, e_scalar, e_declared, e_closure)
    @printf("Stokeslet3D + remainder, direct against Ewald: %.1e\n", e_direct_rem)
    @test e_tensor < 5tol && e_scalar < 5tol && e_declared < 1e-13 && e_closure < 1e-13 && e_direct_rem < 5tol
    @test kernel_shape(T) == (2, 2) && kernel_shape(S) == (1, 1)
    @test_throws ArgumentError Evaluator(r -> G3(r), ad)                              # a tensor closure, no shape
    @test_throws ArgumentError Evaluator(r -> Mat2(1, 0, 0, 2) / norm(r), bd; targets=ad.x[1:3])
    # every built-in kernel passes the probe
    for K in (LaplaceKernel2D(), MHKernel(2.0), LaplaceGradient2D(), BiharmonicGradient2D(), LaplaceKernel3D(),
              Stokeslet3D(), OpenSurfaces.StokesKernel(1.0, 2.0))
        @test Evaluator(K, ad; fast=:direct, targets=ad.x[1:3]) isa Evaluator
    end
    @test Evaluator(OpenSurfaces.TractionKernel(bd.n), ad; fast=:direct, targets=bd.x) isa Evaluator
    @test Evaluator(MHNormalKernel(2.0, bd.n), ad; fast=:direct, targets=bd.x) isa Evaluator
end

@testset "a SymbolKernel as the kernel: Ewald of the whole kernel against its direct sum, tol 1e-6" begin
    # remainders that decay exponentially (the exact pairs) and algebraically (screened kernels, R with a cone at
    # ρ = 0): the remainder split by weight between the long and the short range (src/kernels/ewald.jl)
    tol = 1e-6; p = 10
    bd, ad, _ = build_mesh(discretize_curve(θ -> Vec2(1.5cos(θ), 0.75sin(θ)), p; tol=1e-6), p)
    F = [Vec2(cos(x[2]), x[1]) for x in ad.x]; g = [1 + x[1] - x[2]^2 for x in ad.x]
    κ = 4.0; Δ(k) = (s = sqrt(k^2 + κ^2); -κ^2 / (2k * s * (k + s)))
    Ks = @test_logs (:warn,) match_mode=:any SymbolKernel(k -> 1 / (2sqrt(k^2 + κ^2)); base=LaplaceKernel3D(),
                                                          scale=1 / κ, outputs=(:value,), remainder=Δ, tol)
    Kt = @test_logs (:warn,) match_mode=:any SymbolKernel(k -> (1 / sqrt(k^2 + κ^2), 1 / (2sqrt(k^2 + κ^2)));
                                                          base=Stokeslet3D(), scale=1 / κ, tol)
    for (name, K, f, kw) in (("scalar, algebraic", Ks, g, (; density=:smooth)),
                             ("scalar, algebraic, area + boundary", Ks, g, (; targets=(:area, :boundary), bd)),
                             ("scalar, exponential", scalarK(outputs=(:value,)), g, (;)),
                             ("tensor, algebraic", Kt, F, (;)),
                             ("tensor, algebraic, points", Kt, F, (; targets=ad.x[1:7:end])),
                             ("tensor, exponential", tensorK(), F, (;)))
        E = Evaluator(K, ad; fast=:ewald, tol, kw...)
        @test E.far isa Union{OpenSurfaces._EwaldFar,OpenSurfaces._EwaldLaplaceFar}
        a = E(f); b = Evaluator(K, ad; fast=:direct, tol, kw...)(f)
        e = a isa Tuple ? maximum(relmax(x, y) for (x, y) in zip(a, b)) : relmax(a, b)
        @printf("SymbolKernel %-36s Ewald − direct %.1e\n", name, e)
        @test e < tol
    end
    # a tensor SymbolKernel's (α, β) with β ~ 1/ρ at ρ = 0 (an algebraic remainder): β ρ² smooth, β r rᵀ → 0
    βρ2 = [(ρ = 10.0^-j; kernel_remainder(Kt, ρ)[2] * ρ^2) for j in 2:4]
    @test isapprox(βρ2[2] / βρ2[1], 0.1; rtol=0.05) && isapprox(βρ2[3] / βρ2[2], 0.1; rtol=0.05)
    @test kernel_remainder(Kt, 0.0)[2] == 0
    @test_throws ArgumentError Evaluator(Kt, ad; fast=:fmm)
    # a table that covers the film (diameter 3) but not the long-range window: Ewald refuses it
    Kshort = @test_logs (:warn,) match_mode=:any SymbolKernel(k -> (1 / sqrt(k^2 + κ^2), 1 / (2sqrt(k^2 + κ^2)));
                                                              base=Stokeslet3D(), scale=1 / κ, tol, ρc=3.1)
    @test_throws ArgumentError Evaluator(Kshort, ad)
end

@testset "StokesKernel (G2 + G3) by Ewald against its direct sum, tol 1e-6" begin
    tol = 1e-6; p = 10
    bd, ad, _ = build_mesh(discretize_curve(θ -> Vec2(1.5cos(θ), 0.75sin(θ)), p; tol=1e-6), p)
    F = [Vec2(cos(x[2]), x[1]) for x in ad.x]
    pts = vcat(ad.x[1:5:end], [Vec2(0.1, 0.2)])        # explicit targets, most of them on area nodes
    for (μ2, μ3) in ((1.0, 2.0), (0.3, Inf), (Inf, 2.0)), (name, kw) in (("area", (;)), ("points", (; targets=pts)),
                                                                        ("boundary", (; targets=:boundary, bd)))
        K = OpenSurfaces.StokesKernel(μ2, μ3)
        E = Evaluator(K, ad; fast=:ewald, tol, kw...)          # (:auto takes the direct sum for few targets)
        @test E.far isa OpenSurfaces._EwaldFar
        e = relmax(E(F), Evaluator(K, ad; fast=:direct, tol, kw...)(F))
        @printf("StokesKernel(%s, %s) %-8s Ewald − direct %.1e\n", μ2, μ3, name, e)
        @test e < tol
    end
    @test_throws ArgumentError Evaluator(OpenSurfaces.StokesKernel(1.0, 2.0), ad; fast=:ewald, targets=(:area, :boundary), bd)
    # fast = :auto: Ewald for many targets, the direct sum below OpenSurfaces._direct_targets(eps) (2000 at 1e-6,
    # about 12000 at 1e-10); an explicit fast = :ewald or a plan keeps Ewald
    half = ad.x[1:2:end]                               # 6950 targets: Ewald at eps 1e-6, direct at 1e-10
    @test length(bd.x) < OpenSurfaces._direct_targets(tol) < length(half) < OpenSurfaces._direct_targets(1e-10)
    for K in (Stokeslet3D(), LaplaceKernel3D(), OpenSurfaces.StokesKernel(1.0, 2.0))
        far(; kw...) = Evaluator(K, ad; tol, kw...).far
        @test far(; targets=:boundary, bd) isa OpenSurfaces._DirectFar
        @test !(far(; targets=half) isa OpenSurfaces._DirectFar)
        @test far(; targets=half, eps=1e-10) isa OpenSurfaces._DirectFar
        @test !(far(; targets=:boundary, bd, fast=:ewald) isa OpenSurfaces._DirectFar)
    end
    @test Evaluator(Stokeslet3D(), ad; targets=:boundary, bd, plan=EwaldStokes3D(ad.x, bd.x; tol), tol).far isa
          OpenSurfaces._EwaldFar
    @test_throws ArgumentError Evaluator(OpenSurfaces.StokesKernel(1.0, 2.0), ad; fast=:fmm)
end

@testset "line sources with targets on Γ" begin
    # targets on Γ, at nodes and between them (not NaN). On a circle of radius R: (1/2π) ∫ log|x − y| dΓ_y = R log R and,
    # on the unit circle, of cos 2θ: −cos(2θ)/4; the modified Helmholtz kernel against adaptive quadrature in θ
    p = 10; tol = 1e-6; R = 1.3; k = 2.0
    bdR = BoundaryDisc(discretize_curve(θ -> R * Vec2(cos(θ), sin(θ)), p; tol=1e-8), p)
    bd1 = BoundaryDisc(discretize_curve(θ -> Vec2(cos(θ), sin(θ)), p; tol=1e-8), p)
    on(R, n) = [R * Vec2(cos(t), sin(t)) for t in range(0.013, 2π; length=n)]       # between the nodes
    e1 = maximum(maximum(abs.(Evaluator(LaplaceKernel2D(), bdR; targets=X, tol)(ones(length(bdR))) .- R * log(R)))
             for X in (bdR.x, on(R, 37), [curve_eval(bdR.panels[3], 1.0)[1]]))                  # the last: a panel end
    c2 = [cos(2atan(x[2], x[1])) for x in bd1.x]
    e2 = maximum(maximum(abs.(Evaluator(LaplaceKernel2D(), bd1; targets=X, tol)(c2) .+ [cos(2atan(x[2], x[1])) / 4 for x in X]))
             for X in (bd1.x, on(1.0, 37)))
    mh(x) = (θx = atan(x[2], x[1]);                                    # one period from the target's angle
             quadgk(θ -> besselk(0, k * norm(x - R * Vec2(cos(θ), sin(θ)))) * R / 2π, θx, θx + 2π; rtol=1e-13)[1])
    X = vcat(bdR.x[1:7:end], on(R, 23))
    Em = Evaluator(MHKernel(k), bdR; targets=X, tol); Ed = Evaluator(MHKernel(k), bdR; targets=X, fast=:direct, tol)
    um = Em(ones(length(bdR)))
    e3 = maximum(abs.(um .- mh.(X))) / maximum(abs, um); e3f = relmax(um, Ed(ones(length(bdR))))
    ν = [Vec2(x[2], 1 + x[1]^2) for x in bdR.x]
    e4 = relmax(Evaluator(OpenSurfaces.StokesKernel(1.7, Inf), bdR; targets=bdR.x, tol)(ν),
                deinterleave(slp2d_velocity_matrix(bdR; μ_2D=1.7, tol) * interleave(ν)))
    @printf("line sources on Γ: Laplace R log R %.1e, cos 2θ %.1e; MH against quadgk %.1e (FMM − direct %.1e); 2D Stokeslet at the nodes against slp2d_velocity_matrix %.1e\n", e1, e2, e3, e3f, e4)
    @test max(e1, e2, e3, e4) < 10tol && e3f < 10tol
    @test_throws ArgumentError Evaluator(Stokeslet3D(), bdR; targets=bdR.x[1:3])
    # a kernel bounded at r = 0 (ρ log ρ, 0 there) keeps the adaptive weights on Γ
    kb(ρ) = ρ == 0 ? 0.0 : ρ * log(ρ)
    rb(x) = (θx = atan(x[2], x[1]); quadgk(θ -> kb(norm(x - R * Vec2(cos(θ), sin(θ)))) * R, θx, θx + 2π; rtol=1e-13)[1])
    Xb = vcat(bdR.x[1:5:end], on(R, 17))
    ub = Evaluator(r -> Mat2(kb(norm(r)), 0, 0, 0), bdR; targets=Xb, tol)(ones(length(bdR)))
    e5 = maximum(abs.(ub .- rb.(Xb))) / maximum(abs, ub)
    @printf("line sources on Γ, bounded kernel ρ log ρ (adaptive): %.1e\n", e5)
    @test e5 < 10tol
end

@testset "evaluator_matrix, errors" begin
    p = 10
    bd, ad, _ = build_mesh(discretize_curve(θ -> Vec2(cos(θ), 0.6sin(θ)), p; tol=1e-10), p)
    slot = r -> Mat2(exp(-norm(r)), 0, 0, 0)
    E = Evaluator(slot, ad; targets=bd.x)
    f = [x[1] + 2 for x in ad.x]
    @test evaluator_matrix(E) * f ≈ E(f) rtol = 1e-13
    L = Evaluator(slot, bd; targets=ad.x); σ = [x[2] for x in bd.x]
    @test evaluator_matrix(L) * σ ≈ L(σ) rtol = 1e-13
    @test_throws ArgumentError Evaluator(LaplaceKernel3D(), ad; density=:weighted)
    @test_throws ArgumentError Evaluator(LaplaceKernel3D(), ad; targets=:boundary)
    @test_throws ArgumentError Evaluator(Stokeslet3D(), ad; targets=(:area, :boundary), bd)
    @test_throws ArgumentError evaluator_matrix(Evaluator(LaplaceKernel3D(), ad; targets=(:area, ad.x[1:4])))
    # a per-call eps: the FMM paths take it, the Ewald plan and a remainder's local sum refuse a different one
    F = [Vec2(x[2], 1) for x in ad.x]
    Ew = Evaluator(Stokeslet3D(), ad)
    @test_throws ArgumentError Ew(F; eps=1e-10)
    @test Ew(F; eps=Ew.eps) == Ew(F)
    Rd = SymbolKernel(k -> tanh(0.3k) / (2k); base=LaplaceKernel3D(), scale=0.3, outputs=(:value,),
                      remainder=k -> -1 / (k * (exp(0.6k) + 1)))
    @test_throws ArgumentError Evaluator(LaplaceKernel3D(), ad; density=:smooth, remainder=Rd)(f; eps=1e-9)
    Lf = Evaluator(LaplaceKernel3D(), ad; fast=:fmm)
    @test Lf(f; eps=1e-10) ≈ Lf(f) rtol = 1e-5
    @test_throws ArgumentError Evaluator(LaplaceKernel3D(), ad)(f; eps=1e-10)     # its default is Ewald
end

@testset "operators: evaluator_matrix by shape, mul!, size, local blocks, line traction K(i, r)" begin
    p = 8; tol = 1e-6
    bd, ad, _ = build_mesh(discretize_curve(θ -> Vec2(cos(θ), 0.6sin(θ)), p; tol=1e-4), p)
    N = length(ad); F = [Vec2(x[2] + 1, x[1]^2) for x in ad.x]; g = [1 + x[1] - x[2]^2 for x in ad.x]
    rel(a, b) = norm(a - b) / norm(b)
    e = Dict{String,Float64}()
    # dense matrices against the apply of direct evaluators (to roundoff), by shape and density mode
    E = Evaluator(Stokeslet3D(; μ_3D=1.5), ad; fast=:direct, tol)
    M = evaluator_matrix(E)
    e["tensor"] = rel(M * interleave(F), interleave(E(F)))
    Er = Evaluator(Stokeslet3D(), ad; remainder=tensorK(), fast=:direct, tol)
    e["tensor + remainder"] = rel(evaluator_matrix(Er) * interleave(F), interleave(Er(F)))
    Eg = Evaluator(LaplaceGradient2D(), ad; density=:smooth, targets=:boundary, bd, fast=:direct, tol)
    e["vector, smooth"] = rel(evaluator_matrix(Eg) * g, interleave(Eg(g)))
    Eu = Evaluator(LaplaceKernel3D(), ad; upsampling=AreaUpsampling(ad, 12), fast=:direct, tol)
    e["scalar, upsampled"] = rel(evaluator_matrix(Eu) * g, Eu(g))
    Rd = SymbolKernel(k -> tanh(0.3k) / (2k); base=LaplaceKernel3D(), scale=0.3, outputs=(:value,),
                      remainder=k -> -1 / (k * (exp(0.6k) + 1)))
    El = SingleLayer3D(ad; remainder=Rd).E
    e["scalar, local sum of a remainder"] = rel(evaluator_matrix(El) * g, El(g))
    Ef = Evaluator(Stokeslet3D(; μ_3D=1.5), ad; tol)                      # Ewald: the same operator to eps
    e["tensor, Ewald (to eps)"] = rel(evaluator_matrix(Ef) * interleave(F), interleave(Ef(F)))
    # the linear-operator interface
    y = zeros(2N); mul!(y, E, interleave(F))
    e["mul!"] = rel(y, interleave(E(F)))
    y2 = ones(2N); mul!(y2, E, interleave(F), 2.0, -1.0)
    e["mul! 5-arg"] = rel(y2, 2 .* interleave(E(F)) .- 1)
    # local blocks: the diagonal blocks of the dense matrix
    idx, blocks = local_blocks(E, panel_groups(ad))
    e["local blocks (tensor)"] = maximum(norm(blocks[k] - M[idx[k], idx[k]]) / norm(M[idx[k], idx[k]]) for k in eachindex(idx))
    Es = Evaluator(LaplaceKernel3D(), ad; density=:smooth, fast=:direct, tol); Ms = evaluator_matrix(Es)
    is_, bs = local_blocks(Es, panel_groups(ad; group=false))
    e["local blocks (scalar, smooth)"] = maximum(norm(bs[k] - Ms[is_[k], is_[k]]) / norm(Ms[is_[k], is_[k]]) for k in eachindex(is_))
    # off-curve traction from the corrected sum, against slp2d_traction_eval point by point
    lq = LineQuad(bd; tol)
    X = [Vec2(0.95cos(t), 0.6 * 0.95sin(t)) for t in range(0, 2π; length=31)][1:30]
    nX = [normalize(Vec2(cos(t) / 1, sin(t) / 0.6)) for t in range(0, 2π; length=31)][1:30]
    ν = [Vec2(x[1], 1 - x[2]) for x in bd.x]
    Et = Evaluator(OpenSurfaces.TractionKernel(nX), bd; targets=X, fast=:direct, tol)
    e["line traction K(i, r)"] = rel(interleave(Et(ν)), interleave([slp2d_traction_eval(lq, X[i], nX[i], ν) for i in eachindex(X)]))
    for (k, v) in sort(collect(e)); @printf("operators: %-34s %.1e\n", k, v); end
    @test all(v < 1e-12 for (k, v) in e if k != "tensor, Ewald (to eps)") && e["tensor, Ewald (to eps)"] < 10tol
    @test size(E) == (2N, 2N) && size(Es) == (N, N) && size(Eg) == (2length(bd), N) && size(Eu) == (N, N)
    @test E * interleave(F) ≈ interleave(E(F)) rtol = 1e-14
    @test_throws DimensionMismatch E * ones(N)
    @test_throws ArgumentError local_blocks(Eg, panel_groups(ad))
end
