# Kernels from Fourier symbols (src/kernels/symbol.jl) against exact Hankel pairs, c = 2H:
#   e^{−ck}/k ↔ 1/(2π√(ρ² + c²)) (value, gradient, Δ⁻¹ gradient, Δ⁻¹ potential), and its J2 transform for a tensor;
# a zero remainder gives the base exactly; the regularized gradient against adaptive quadrature; the errors (no base,
# beyond the table without a far field) and the far field; SmoothConvolution of the scalar remainder against the
# direct sum. The pairs are in common/symbol_pairs.jl; the Ewald sums with a remainder are test_kernels.jl's.
#   julia -t 8 --project -e 'using Pkg; Pkg.test(test_args=["symbol"])'   (from src/OpenSurfaces)
include(joinpath(@__DIR__, "common", "symbol_pairs.jl"))
using Printf, Test, Random, QuadGK
using OpenSurfaces: _x2mJ1

@testset "scalar and tensor tables against exact pairs" begin
    K = scalarK(outputs=(:value, :gradient, :potential_gradient, :potential))
    ρs = (0.0, 0.01, 0.3, 2.0, 9.0, 14.9)
    sc = [maximum(abs(exact_scalar(ρ)[j]) for ρ in ρs) for j in 1:4]
    es = maximum(maximum(abs.(kernel_remainder(K, ρ) .- exact_scalar(ρ)) ./ sc) for ρ in ρs)
    T = tensorK(); M = Stokeslet3D()
    et = maximum(begin
        (α, β), (α0, β0) = kernel_remainder(T, ρ), exact_tensor(ρ)
        max(abs(α - α0) * c, abs(β - β0) * c^3)
    end for ρ in ρs)
    r = Vec2(0.7, -0.4); α0, β0 = exact_tensor(norm(r))
    ek = norm(T(r) - (M(r) + α0 * I + β0 * r * r')) * c
    eb = abs(K(r) - (1 / (4π * norm(r)) + exact_scalar(norm(r))[1])) * c
    @printf("symbol kernels against exact pairs: scalar outputs %.1e, tensor %.1e, K(r) %.1e (tensor), %.1e (scalar)\n",
            es, et, ek, eb)
    @test es < 1e-12 && et < 1e-12 && ek < 1e-12 && eb < 1e-12
end

@testset "zero remainder, errors, far field" begin
    Z = SymbolKernel(k -> kernel_symbol(Stokeslet3D(; μ_3D=2.0), k); base=Stokeslet3D(; μ_3D=2.0), scale=1.0)
    r = Vec2(0.3, 1.1)
    @test Z(r) == G3(r; μ_3D=2.0)
    Zs = SymbolKernel(k -> 1 / (2k); base=LaplaceKernel3D(), scale=1.0)
    @test Zs(r) == 1 / (4π * norm(r))
    @test_throws ArgumentError SymbolKernel(k -> 1 / k; scale=1.0)
    @test_throws DomainError kernel_remainder(scalarK(), 15.1)
    F = scalarK(outputs=(:value,), far=ρ -> (exact_scalar(ρ)[1],))
    @test kernel_remainder(F, 20.0)[1] == exact_scalar(20.0)[1] && table_range(F) == 30H
    D = @test_logs (:warn,) SymbolKernel(k -> exp(-c * k); base=LaplaceKernel3D(), scale=H,   # R ~ −1/(4πρ): warned
                                         remainder=k -> exp(-c * k) - 1 / (2k))
    @test kernel_integral(D) ≈ 1                                    # ∫ K dA = ŝ(0) of a decaying kernel
end

# a 2D Stokeslet base with a symbol (G2 has no kernel_symbol method); a base matching a membrane kernel at both ends
struct Stokeslet2D; η::Float64; end
(K::Stokeslet2D)(r::Vec2) = G2(r; μ_2D=K.η)
OpenSurfaces.kernel_symbol(K::Stokeslet2D, k) = (1 / (K.η * k^2), 0.0)
struct DeepMembrane; η::Float64; end
(K::DeepMembrane)(r::Vec2) = zero(Mat2)
OpenSurfaces.kernel_symbol(K::DeepMembrane, k) = (1 / (K.η * k^2 + 2k), 1 / (2k))

@testset "remainders not integrable at k = 0 are refused" begin
    # Saffman–Delbrück, (I − k̂k̂ᵀ)/(ηk² + 2μk), on a G2 base: Δ ~ −1/(ηk²) at small k; accepting it would build a
    # table holding a constant set by the k grid (trace −0.618 at dk = 0.25 against 0.12)
    η, μ = 1.0, 0.5; ℓ = η / (2μ)
    sd(k) = (1 / (η * k^2 + 2μ * k), 0.0)
    for kw in ((;), (; dk=0.05 / ℓ), (; kbreaks=[0.0; 2.0 .^ (-20:0) ./ ℓ; (2:1:25) ./ ℓ]))
        @test_throws ArgumentError SymbolKernel(sd; base=Stokeslet2D(η), scale=ℓ, kw...)
    end
    @test_throws ArgumentError SymbolKernel(k -> 1 / (2k) + 1 / k^2; base=LaplaceKernel3D(), scale=1.0)
    # a base matching at small k too (a finite-depth membrane over its deep kernel): k Δ is bounded, so it builds
    H = 0.7
    @test SymbolKernel(k -> (1 / (η * k^2 + 2k * coth(k * H)), tanh(k * H) / (2k)); base=DeepMembrane(η), scale=H) isa SymbolKernel
    # Δ ~ k⁻⁴ with a regularized gradient builds (k⁵ Δ → 0); its plain gradient does not (k³ Δ ~ 1/k)
    @test SymbolKernel(k -> 0.0; base=LaplaceKernel3D(), scale=H, outputs=(:gradient,), regularize=true,
                       remainder=k -> -exp(-k) / k^4) isa SymbolKernel
    @test_throws ArgumentError SymbolKernel(k -> 0.0; base=LaplaceKernel3D(), scale=H, outputs=(:gradient,),
                                            remainder=k -> -exp(-k) / k^4)
end

@testset "regularized gradient (remainder not integrable at k = 0)" begin
    # Δ = −e^{−ck}/k⁴: ∇ of its transform up to a multiple of r, (1/2πρ) ∫ Δ k² (kρ/2 − J1(kρ)) dk
    K = SymbolKernel(k -> 0.0; base=LaplaceKernel3D(), scale=H, outputs=(:gradient,), regularize=true,
                     remainder=k -> -exp(-c * k) / k^4)
    gref(ρ) = quadgk(k -> -exp(-c * k) / k^2 * _x2mJ1(k * ρ), 0, 1, 10, 60; rtol=1e-13)[1] / (2π * ρ)
    e = maximum(abs(kernel_remainder(K, ρ)[1] - gref(ρ)) for ρ in (0.05, 0.9, 4.2, 11.3))
    @printf("regularized gradient against adaptive quadrature: %.1e\n", e)
    @test e < 1e-12
end

@testset "SmoothConvolution of the scalar remainder, tol 1e-6" begin
    rng = MersenneTwister(8); tol = 1e-6
    src = [Vec2(3rand(rng) - 1.5, 1.5rand(rng) - 0.75) for _ in 1:1500]
    trg = [Vec2(2rand(rng) - 1, rand(rng) - 0.5) for _ in 1:300]
    g = randn(rng, length(src)); v(r) = exact_scalar(norm(r))[1]
    sref = [sum(v(x - y) * gj for (y, gj) in zip(src, g)) for x in trg]
    S = SmoothConvolution(r -> (v(r),), 1, 1, src, trg; scale=H, tol)
    eS = norm(smooth_apply(S, reshape(g, :, 1))[:, 1] - sref) / norm(sref)
    @printf("SmoothConvolution of the scalar remainder: %.1e\n", eS)
    @test eS < 10tol
end
