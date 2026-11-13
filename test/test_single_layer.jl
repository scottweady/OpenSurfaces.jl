# The 3D Laplace single layer S_Ω (src/evaluators/single_layer.jl) on the unit disc against the closed
# forms of Gomez Herrera & Weady (2026), eq. 4.6: S_D[1] = E(r)/π and S_D[ω ζ^j] = ((λ_j^j − λ_{j+1}^{j+1}|ζ|²)/8) ζ^j,
# ω = √(1 − r²), for the real densities Re ω ζ^j, j = 0–2, at the area nodes.
#   julia -t 8 --project -e 'using Pkg; Pkg.test(test_args=["single_layer"])'   (from the package root)
using OpenSurfaces, LinearAlgebra, Printf, Test
using SpecialFunctions: gamma, ellipe

@testset "3D Laplace single layer on the disc (GH&W eq. 4.6)" begin
    λ(l, m) = gamma((l + m) / 2 + 0.5) / gamma((l + m) / 2 + 1) *
              gamma((l - m) / 2 + 0.5) / gamma((l - m) / 2 + 1)
    pans = discretize_curve(θ -> Vec2(cos(θ), sin(θ)), 10; tol=1e-6)
    _, ad, _ = build_mesh(pans, 10; edge_layers=2)
    L = SingleLayer3D(ad; tol=1e-7, eps=1e-7)
    r = norm.(ad.x); ω = sqrt.(max.(1 .- r .^ 2, 0.0))
    e1 = maximum(abs.(single_layer(L, ones(length(ad))) .- ellipe.(r .^ 2) ./ π))
    ej = maximum(begin
        ζj = [real(complex(x[1], x[2])^j) for x in ad.x]
        maximum(abs.(single_layer(L, ω .* ζj) .- (λ(j, j) .- λ(j + 1, j + 1) .* r .^ 2) ./ 8 .* ζj))
    end for j in 0:2)
    @printf("single layer on the disc: S_D[1] %.1e, S_D[ω ζ^j] %.1e\n", e1, ej)
    @test e1 < 1e-5 && ej < 1e-5
end

@testset "a remainder that decays algebraically: the screened kernel e^{−κρ}/(4πρ)" begin
    # symbol 1/(2√(k² + κ²)) on LaplaceKernel3D: Δ ~ −κ²/(4k³), so ℓ = (e^{−κρ} − 1)/(4πρ) has a cone at ρ = 0. The
    # smooth rule integrated it to 4e-5 whatever tol (the base's corrections only); the whole kernel's corrections now.
    # S_κ[1] on the unit disc against polar rays: (1/4π) ∫ (1 − e^{−κρmax(φ)})/κ dφ (trapezoid, spectral)
    κ = 4.0
    Δ(k) = (s = sqrt(k^2 + κ^2); -κ^2 / (2k * s * (k + s)))
    ref(x; M=4096) = sum(begin
        e = Vec2(cos(φ), sin(φ)); b = dot(x, e); ρm = -b + sqrt(b^2 - dot(x, x) + 1)
        -expm1(-κ * ρm) / κ
    end for φ in 2π .* (0:M-1) ./ M) * (2π / M) / 4π
    _, ad, _ = build_mesh(discretize_curve(θ -> Vec2(cos(θ), sin(θ)), 10; tol=1e-6), 10)
    idx = 1:7:length(ad); r = ref.(ad.x[idx])
    errs = map((1e-6, 1e-8)) do tol
        K = @test_logs (:warn,) match_mode=:any SymbolKernel(k -> 1 / (2sqrt(k^2 + κ^2)); base=LaplaceKernel3D(),
                                                            scale=1 / κ, outputs=(:value,), remainder=Δ, tol)
        @test K.decay === :algebraic
        u = single_layer(SingleLayer3D(ad; tol, remainder=K), ones(length(ad)))
        maximum(abs.(u[idx] .- r)) / maximum(abs, r)
    end
    @printf("screened single layer S_κ[1] on the disc against polar rays: %.1e (tol 1e-6), %.1e (tol 1e-8)\n", errs...)
    @test errs[1] < 1e-6 && errs[2] < 1e-7
    # an exponentially decaying remainder keeps its table and its base-only corrections; Ewald refuses an algebraic one
    H = 0.5
    D = SymbolKernel(k -> tanh(k * H) / (2k); base=LaplaceKernel3D(), scale=H, remainder=k -> -1 / (k * (exp(2k * H) + 1)))
    @test D.decay === :exponential
    T = @test_logs (:warn,) match_mode=:any SymbolKernel(k -> (1 / sqrt(k^2 + κ^2), 1 / (2sqrt(k^2 + κ^2)));
                                                         base=Stokeslet3D(), scale=1 / κ)
    @test_throws ArgumentError EwaldStokes3D(ad.x; remainder=T)
end
