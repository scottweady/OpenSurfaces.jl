# A kernel from its Fourier symbol: the screened kernel e^{−κρ}/(4πρ), symbol 1/(2√(k² + κ²)), as a `SymbolKernel`
# on the base `LaplaceKernel3D()` (symbol 1/(2k)): the remainder ℓ(ρ) = (e^{−κρ} − 1)/(4πρ) is tabulated by a
# Hankel transform of Δ = 1/(2√(k² + κ²)) − 1/(2k). Its table against the closed form, then S_κ[1] on the unit disc
# against the polar-ray reference (1/4π) ∫ (1 − e^{−κ ρmax(φ)})/κ dφ (trapezoid rule in φ, spectrally accurate).
# Δ decays algebraically (≈ −κ²/(4k³)), so the constructor extends its k grid (64× at most) and warns that the
# tail beyond it may leave the table off near ρ = 0; the table's actual error is printed below.
#   julia -t 8 --project=. examples/screened_kernel.jl          (from the package root)
using OpenSurfaces, LinearAlgebra, Printf

κ = 4.0
Δ(k) = (s = sqrt(k^2 + κ^2); -κ^2 / (2k * s * (k + s)))         # the remainder without cancellation at large k
K = SymbolKernel(k -> 1 / (2sqrt(k^2 + κ^2)); base=LaplaceKernel3D(), scale=1 / κ, outputs=(:value,), remainder=Δ)
ℓ(ρ) = ρ == 0 ? -κ / 4π : expm1(-κ * ρ) / (4π * ρ)
ρs = range(0, 0.99table_range(K), length=2000)
@printf("table of ℓ on [0, %.2f]: max error %.1e (relative to |ℓ(0)| = κ/4π), %.1e for ρ ≥ 0.1\n", table_range(K),
        maximum(abs(kernel_remainder(K, ρ)[1] - ℓ(ρ)) for ρ in ρs) / (κ / 4π),
        maximum(abs(kernel_remainder(K, ρ)[1] - ℓ(ρ)) for ρ in ρs if ρ >= 0.1) / (κ / 4π))
@printf("K(r) against e^{−κρ}/(4πρ): max relative error %.1e\n",
        maximum(abs(K(Vec2(ρ, 0)) * 4π * ρ * exp(κ * ρ) - 1) for ρ in ρs[2:end] if ρ < 2))

# S_κ[1] on the unit disc: the single layer of the whole kernel, base + remainder
ref(x; M=4096) = sum(begin
    e = Vec2(cos(φ), sin(φ)); b = dot(x, e); ρm = -b + sqrt(b^2 - dot(x, x) + 1)
    -expm1(-κ * ρm) / κ
end for φ in 2π .* (0:M-1) ./ M) * (2π / M) / 4π
_, ad, _ = build_mesh(discretize_curve(θ -> Vec2(cos(θ), sin(θ)), 10), 10)
idx = 1:7:length(ad); u0 = ref.(ad.x[idx])
for tol in (1e-6, 1e-8)
    u = single_layer(SingleLayer3D(ad; tol, remainder=K), ones(length(ad)))
    @printf("S_κ[1] on the disc (tol %.0e): max relative error against polar rays %.1e\n", tol,
            maximum(abs.(u[idx] .- u0)) / maximum(abs, u0))
end
