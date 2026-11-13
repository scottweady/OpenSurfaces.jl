# Shared by test_symbol.jl (the tables) and test_kernels.jl (Ewald sums with a remainder): the exact Hankel pair
# e^{−ck}/k ↔ 1/(2π√(ρ² + c²)), c = 2H, as a scalar remainder and as a tensor one, the `SymbolKernel`s built from
# them, and the direct sum of the tensor remainder.

using OpenSurfaces, LinearAlgebra, StaticArrays

const H = 0.5
const c = 2H
# the remainder e^{−ck}/k in closed form (outputs :value, :gradient, :potential_gradient, :potential)
q_(ρ) = sqrt(ρ^2 + c^2)
# (q − c = ρ²/(q + c), stable at ρ = 0)
exact_scalar(ρ) = (1 / (2π * q_(ρ)), -1 / (2π * q_(ρ)^3), 1 / (2π * (q_(ρ) + c)),
                   (ρ^2 / (q_(ρ) + c) - c * log((c + q_(ρ)) / (2c))) / 2π)
# the tensor remainder (0, e^{−ck}/k): s = d = e^{−ck}/(2k); A0 = (1/2) value, A2 = (1/4π)(q − c)²/(ρ² q)
function exact_tensor(ρ)
    A0 = 1 / (4π * q_(ρ)); A2r = 1 / (4π * (q_(ρ) + c)^2 * q_(ρ))      # A2/ρ²
    (A0 + ρ^2 * A2r, -2A2r)
end
scalarK(; kw...) = SymbolKernel(k -> 1 / (2k) + exp(-c * k) / k; base=LaplaceKernel3D(), scale=H, kw...)
tensorK(; kw...) = SymbolKernel(k -> (1 / k, 1 / (2k) + exp(-c * k) / k); base=Stokeslet3D(), scale=H, kw...)

"""Σ_j R(x − y_j) f_j at the targets, R the exact tensor remainder at μ = 1 (all pairs, i = j included)."""
function rem_sum(y, f, x)
    R(r) = ((α, β) = exact_tensor(norm(r)); α * I + β * r * r')
    [sum(Vec2(R(xi - yj) * fj) for (yj, fj) in zip(y, f)) for xi in x]
end
