# Modified Helmholtz kernel G = K₀(kρ)/2π ((k² − Δ)G = δ) and its normal derivative at the target, for the
# potentials of a modified Helmholtz problem on the film. Far field:
# FMM2D's Helmholtz FMM at the imaginary wavenumber ik, which returns (i/4)H₀⁽¹⁾(ikρ) = K₀(kρ)/2π.

# Table of (eᶻK₀(z), eᶻzK₁(z)): Chebyshev (degree 19) on the doubling
# panels [z₀2ʲ, z₀2ʲ⁺¹], z₀ = 1e-8, up to 2^37 z₀ ≈ 1.4e3 (log z and z² log z are smooth on a panel relative to its
# width). Below z₀: K₀ = −log(z/2) − γ_E, zK₁ = 1 (error O(z² log z)); above: 0 (e^{−z} < 1e-596).
# SpecialFunctions' besselk costs ~520 ns, which made the near corrections of V_k three times those of G2.
const _MH_Z0 = 1e-8
const _MH_NP = 37
const _MH_NC = 20
function _mh_table()
    nc = _MH_NC
    t = [cos(π * (j - 0.5) / nc) for j in 1:nc]
    Ti = inv([cos((m - 1) * π * (j - 0.5) / nc) for j in 1:nc, m in 1:nc])
    c = zeros(nc, _MH_NP, 2)
    for p in 1:_MH_NP
        a = _MH_Z0 * 2.0^(p - 1)
        z = [a * (1 + (tj + 1) / 2) for tj in t]
        c[:, p, 1] = Ti * SpecialFunctions.besselkx.(0, z); c[:, p, 2] = Ti * (z .* SpecialFunctions.besselkx.(1, z))
    end
    c
end
const _MH_TABLE = _mh_table()
const _MH_ZMAX = _MH_Z0 * 2.0^_MH_NP

"""(K₀(z), zK₁(z)) from the table."""
@inline function _mhK01(z::Float64)
    z < _MH_Z0 && return (-log(z / 2) - Base.MathConstants.eulergamma, 1.0)
    z >= _MH_ZMAX && return (0.0, 0.0)
    p = clamp(exponent(z / _MH_Z0) + 1, 1, _MH_NP)
    a = _MH_Z0 * 2.0^(p - 1)
    x = 2 * (z / a - 1) - 1
    b1 = 0.0; b2 = 0.0; d1 = 0.0; d2 = 0.0
    @inbounds for m in _MH_NC:-1:2
        b1, b2 = 2x * b1 - b2 + _MH_TABLE[m, p, 1], b1
        d1, d2 = 2x * d1 - d2 + _MH_TABLE[m, p, 2], d1
    end
    e = exp(-z)
    @inbounds ((x * b1 - b2 + _MH_TABLE[1, p, 1]) * e, (x * d1 - d2 + _MH_TABLE[1, p, 2]) * e)
end

@inline _mhG(k, ρ) = _mhK01(k * ρ)[1] / 2π
"""Scalar kernel G = K₀(kρ)/2π (as the 2×2 kernel diag(G, 0) of the quadrature routines)."""
struct MHKernel
    k::Float64
end
@inline (K::MHKernel)(r::Vec2) = (ρ = norm(r); ρ == 0 ? zero(Mat2) : Mat2(_mhG(K.k, ρ), 0, 0, 0))
@inline (K::MHKernel)(i::Int, r::Vec2) = K(r)

"""Normal derivative at the target, ∂_{n_x}G = −k K₁(kρ)(r·n_x)/(2πρ)."""
struct MHNormalKernel
    k::Float64
    n::Vector{Vec2}
end
@inline function (K::MHNormalKernel)(i::Int, r::Vec2)
    ρ = norm(r)
    ρ == 0 && return zero(Mat2)
    Mat2(-_mhK01(K.k * ρ)[2] * dot(r, K.n[i]) / (2π * ρ^2), 0, 0, 0)
end
