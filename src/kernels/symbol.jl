# Kernels from Fourier symbols. A radially symmetric kernel on the plane is
# given by its symbol ŝ(k), k = |k|, or a tensor kernel by isotropic mobilities, Ĝ = a(I − k̂k̂ᵀ) + b k̂k̂ᵀ. The symbol
# is written as that of a standard kernel `base` with the same large-k behavior plus a remainder Δ that decays in k,
# so the remainder is smooth in real space and is tabulated by Hankel transforms (x = kρ):
#   scalar   R = (1/2π) ∫ Δ J0(x) k dk                      `:value`
#            R′/ρ = −(1/2π) ∫ Δ k² J1(x)/x k dk             `:gradient`            (∇R = (R′/ρ) r)
#            P′/ρ = (1/2π) ∫ Δ J1(x)/x k dk                 `:potential_gradient`  (P = Δ⁻¹R, ΔP = R)
#            P(ρ) − P(0) = (1/2π) ∫ Δ (1 − J0(x))/k² k dk   `:potential`
#   tensor   R = αI + β r rᵀ,  α = A0 + A2,  β = −2A2/ρ²,  A0 = (1/2π) ∫ s J0(x) k dk,  A2 = (1/2π) ∫ d J2(x) k dk,
#            s = (Δa + Δb)/2,  d = (Δb − Δa)/2.
# The k integrals are Gauss–Legendre on panels (uniform on [0, kmax] by default, or given breakpoints, e.g. dyadic
# toward k = 0 when Δ has structure there); the remainder is tabulated as piecewise Chebyshev (degree nq − 1) on
# panels of width ≈ `scale` up to ρc, with a closed-form far field beyond if one is given. The numerics keep the
# standard kernel for everything singular (near corrections, Ewald short range, self terms, preconditioners) and add
# only the smooth remainder (Ewald long-range multipliers, `SmoothConvolution`, local sums).
# A remainder whose symbol is not integrable at k = 0 (Δ ~ k⁻⁴, the buoyancy kernel) has a gradient only up to a
# multiple of r: `regularize = true` subtracts the x → 0 value of J1(x)/x, and the base must fix that multiple.
# Extracting `base` from the symbol's large-k expansion is not implemented (it needs analysis first): it is given.

# Nyquist wavenumber × scale of a grid that resolves a remainder (≳ 17 for 1e-12; the Ewald plans and
# `SmoothConvolution`)
const _NYQ_H = 20.0

"""Radial functions tabulated as piecewise Chebyshev on [0, ρc] (N values per ρ; beyond ρc the last panel)."""
struct RadialTable{N}
    ρc::Float64
    hp::Float64
    c::Array{Float64,3}            # (nq, panels, N)
end

function RadialTable{N}(f, ρc::Real, npan::Int, nq::Int) where {N}
    hp = ρc / npan
    t = [cos(π * (j - 0.5) / nq) for j in 1:nq]                            # Chebyshev points (1st kind)
    Ti = inv([cos((m - 1) * π * (j - 0.5) / nq) for j in 1:nq, m in 1:nq])
    c = zeros(nq, npan, N)
    Threads.@threads for p in 1:npan
        vals = [f((p - 1 + (tj + 1) / 2) * hp) for tj in t]
        for n in 1:N
            c[:, p, n] = Ti * [v[n] for v in vals]
        end
    end
    RadialTable{N}(Float64(ρc), hp, c)
end

function (T::RadialTable{N})(ρ) where {N}
    p = min(floor(Int, ρ / T.hp), size(T.c, 2) - 1); t = 2(ρ / T.hp - p) - 1
    ntuple(Val(N)) do n
        v = 0.0; T0 = 1.0; T1 = t
        @inbounds for m in 1:size(T.c, 1)
            Tm = m == 1 ? T0 : m == 2 ? T1 : (Tn = 2t * T1 - T0; T0 = T1; T1 = Tn; Tn)
            v += T.c[m, p+1, n] * Tm
        end
        v
    end
end

# ---------------------------------------------------------------------------------------
# Standard kernels with known symbols (at the viscosity they carry; `kernel_symbol(K, k)`)
# ---------------------------------------------------------------------------------------

"""
    kernel_symbol(K, k)

Fourier symbol of the kernel `K` at wavenumber `k`: a scalar, or the mobilities `(a, b)` of a tensor kernel
(Ĝ = a(I − k̂k̂ᵀ) + b k̂k̂ᵀ). Model kernels used as a `SymbolKernel` base add a method.
"""
function kernel_symbol end

"""
    Stokeslet3D(; μ_3D=1)

The in-plane (2×2) block, on the plane z = 0, of the Stokeslet of a **half-space** of viscosity μ = `μ_3D` below a
free surface at z = 0 (a force on the surface, the fluid below it):

    G(r) = (1/(4πμ)) (I/ρ + r rᵀ/ρ³),   symbol (1/(μk), 1/(2μk)),

`G3(r; μ_3D = μ)`. This is the full-space Stokeslet at viscosity μ/2 (twice the Oseen tensor at μ): for an
unbounded fluid of viscosity μ_f, pass μ_3D = 2μ_f.
"""
struct Stokeslet3D
    μ::Float64
end
Stokeslet3D(; μ_3D::Real=1.0) = Stokeslet3D(Float64(μ_3D))
@inline (K::Stokeslet3D)(r::Vec2) = G3(r; μ_3D=K.μ)
kernel_symbol(K::Stokeslet3D, k) = (1 / (K.μ * k), 1 / (2K.μ * k))

"""3D Laplace kernel on the plane, 1/(4πρ); symbol 1/(2k)."""
struct LaplaceKernel3D end
@inline (::LaplaceKernel3D)(r::Vec2) = 1 / (4π * norm(r))
kernel_symbol(::LaplaceKernel3D, k) = 1 / (2k)

# ---------------------------------------------------------------------------------------
# Hankel transforms of k Δ at ρ (k, w nodes and weights)
# ---------------------------------------------------------------------------------------

_j1z(z) = z < 1e-4 ? 0.5 - z^2 / 16 : SpecialFunctions.besselj1(z) / z     # J1(z)/z

# z/2 − J1(z), by its series below z = 0.1 (no cancellation)
_x2mJ1(z) = z < 0.1 ? z^3 / 16 * (1 - z^2 / 24 * (1 - z^2 / 48 * (1 - z^2 / 80))) : z / 2 - SpecialFunctions.besselj1(z)

# (1 − J₀(z))/z², by its series below z = 0.5 (no cancellation; truncation < 1e-15 relative)
function _omj0z2(z)
    z >= 0.5 && return (1 - SpecialFunctions.besselj0(z)) / z^2
    s = z^2; acc = 0.0; t = 0.25
    for j in 1:7
        acc += t; t *= -s / (4 * (j + 1)^2)
    end
    acc
end

# (α, β) at ρ by direct Hankel quadrature (k, w nodes and weights; s, d the symbol combinations times k there)
function _rh_direct(k, w, s, d, ρ)
    A0 = 0.0; A2r = 0.0
    @inbounds for i in eachindex(k)
        z = k[i] * ρ
        A0 += w[i] * s[i] * SpecialFunctions.besselj0(z)
        A2r += w[i] * d[i] * (z < 1e-4 ? k[i]^2 / 8 * (1 - z^2 / 12) : SpecialFunctions.besselj(2, z) / ρ^2)
    end
    A0 /= 2π; A2r /= 2π
    (A0 + ρ^2 * A2r, -2A2r)
end

# scalar outputs at ρ; f = k Δ at the nodes; `reg`: the gradient with J1(x)/x − 1/2
function _scalar_direct(outputs::NTuple{N,Symbol}, k, w, f, ρ, reg::Bool=false) where {N}
    v = 0.0; g = 0.0; pg = 0.0; p = 0.0
    @inbounds for i in eachindex(k)
        z = k[i] * ρ
        :value in outputs && (v += w[i] * f[i] * SpecialFunctions.besselj0(z))
        if reg
            g += w[i] * f[i] * k[i]^2 * (z == 0 ? 0.0 : _x2mJ1(z) / z)
        elseif :gradient in outputs || :potential_gradient in outputs
            j1 = _j1z(z)
            g -= w[i] * f[i] * k[i]^2 * j1
            pg += w[i] * f[i] * j1
        end
        :potential in outputs && (p += w[i] * f[i] * ρ^2 * _omj0z2(z))
    end
    ntuple(n -> (o = outputs[n]; (o === :value ? v : o === :gradient ? g : o === :potential_gradient ? pg : p) / 2π), Val(N))
end

# Gauss–Legendre nodes and weights on [0, kmax] in panels of dk, or on the panels between `kbreaks`
function _symbol_kgrid(q, kmax, dk, kbreaks)
    r = gl_rule(q)
    if kbreaks === nothing
        np = ceil(Int, kmax / dk - 1e-9)
        k = [(p + (r.x[i] + 1) / 2) * dk for p in 0:np-1 for i in 1:q]
        w = [r.w[i] * dk / 2 for p in 0:np-1 for i in 1:q]
        return k, w
    end
    k = Float64[]; w = Float64[]
    for i in 1:length(kbreaks)-1
        a, b = kbreaks[i], kbreaks[i+1]
        append!(k, a .+ (r.x .+ 1) .* ((b - a) / 2)); append!(w, r.w .* ((b - a) / 2))
    end
    k, w
end

# ---------------------------------------------------------------------------------------
# SymbolKernel
# ---------------------------------------------------------------------------------------

"""
    SymbolKernel(symbol; base, scale, far=nothing, remainder=nothing, outputs=nothing, ρc=30scale,
                 nq=16, q=20, kmax=25/scale, dk=0.25/scale, kbreaks=nothing, regularize=false, tol=DEFAULT_TOL)

The kernel of a radial Fourier symbol, `base` plus a smooth remainder tabulated by Hankel transforms (head of
src/kernels/symbol.jl).
- `symbol(k)`: a scalar ŝ(k), or the mobilities `(a(k), b(k))` of an isotropic tensor kernel.
- `base`: a standard kernel (`Stokeslet3D`, `LaplaceKernel3D`, or a kernel with a `kernel_symbol` method) that
  matches the symbol in two limits: at large k (the same singularity, so the remainder Δ decays and is smooth in
  real space) and at small k (k Δ integrable at k = 0, so the Hankel integrals converge; for a gradient output only,
  k³ Δ). The constructor throws when the remainder's integrals diverge at k = 0. Required: extracting the base from
  the symbol is not implemented.
- `scale`: the length on which the remainder varies (the depth H); it sets the panels and the k grid and tells
  the fast sums how fine a grid resolves the remainder.
- `far(ρ)`: the remainder's outputs in closed form beyond ρc. Without it, evaluating beyond ρc is an error; callers
  that switch to their own far field read `ρc`.
- `remainder(k)`: Δ = symbol − base symbol in a numerically stable form (default: the difference).
- `outputs`: for a scalar symbol, a tuple of `:value`, `:gradient`, `:potential_gradient`, `:potential` (default
  `(:value, :gradient)`); a tensor symbol gives (α, β).
- `regularize`: the remainder's symbol is not integrable at k = 0 (head of the file); only `:gradient`, defined up
  to a multiple of r that the base fixes.
- `ρc`, `nq` (Chebyshev points per panel), `q` (Gauss–Legendre points per k panel), `kmax`, `dk`, `kbreaks`: the
  table. `kmax` defaults to 25/scale, which suits a remainder that decays exponentially (as a finite depth's).
- `tol`: for a remainder that decays algebraically (`K.decay === :algebraic`; then R has a cone or a log at ρ = 0),
  the default `kmax` grows (up to 64 ×) until the tail beyond it is below `tol` of |R(0)|, with a warning if it
  cannot. An `Evaluator` integrates such a remainder with the near corrections of the whole kernel, base + R; the
  Ewald sums refuse it (their long-range multiplier would sample the cone).

Callable as `K(r)`: `base(r)` plus the remainder (a `Mat2` for a tensor symbol, a scalar otherwise). `kernel_remainder(K, ρ)`
gives the remainder's outputs alone.
"""
struct SymbolKernel{S,B,N,F}
    symbol::S
    base::B
    scale::Float64
    tensor::Bool
    outputs::NTuple{N,Symbol}
    table::RadialTable{N}
    far::F
    decay::Symbol                  # :exponential (Δ negligible or decaying fast at kmax) or :algebraic
end

function SymbolKernel(symbol; base=nothing, scale::Real, far=nothing, remainder=nothing, outputs=nothing,
                      ρc::Real=30scale, nq::Int=16, q::Int=20, kmax=nothing, dk::Real=0.25 / scale,
                      kbreaks=nothing, regularize::Bool=false, tol::Real=DEFAULT_TOL)
    base === nothing && throw(ArgumentError("SymbolKernel: extracting the singular part of a symbol from its large-k " *
                                            "expansion is not implemented; pass `base`"))
    Δ = remainder !== nothing ? remainder : (k -> _symbol_minus(symbol(k), kernel_symbol(base, k)))
    kmx = kmax === nothing ? 25 / scale : Float64(kmax)
    k, w = _symbol_kgrid(q, kmx, Float64(dk), kbreaks)
    v = [Δ(x) for x in k]
    tensor = v[1] isa Tuple
    outs0 = tensor ? (:α, :β) : outputs === nothing ? (:value, :gradient) : Tuple(outputs)
    _check_small_k(Δ, k, v, Float64(scale), tensor, outs0, regularize)
    # the tail beyond the grid's end: an algebraic one leaves R a cone (or log) at ρ = 0, and the table's R(0) short
    # by the tail; without a given kmax, the grid is extended (up to 64 ×) until the tail is below tol |R(0)|
    kend = kbreaks === nothing ? kmx : Float64(kbreaks[end])
    decay, m, tail, R0 = _tail_decay(Δ, k, w, v, kend, tensor)
    if decay === :algebraic && kmax === nothing && kbreaks === nothing && tail > tol * R0 && m > 1.05
        kmx = min(64, (tail / (tol * R0))^(1 / (m - 1))) * kmx; kmx = dk * ceil(kmx / dk)   # (m ≤ 1: no finite tail)
        k, w = _symbol_kgrid(q, kmx, Float64(dk), nothing)
        v = [Δ(x) for x in k]
        decay, m, tail, R0 = _tail_decay(Δ, k, w, v, kmx, tensor)
    end
    decay === :algebraic && tail > tol * R0 &&
        @warn "SymbolKernel: the remainder decays algebraically (k Δ ~ k^-$(round(m; digits=1))): its tail beyond " *
              "kmax = $(round(kmx * scale; sigdigits=3))/scale is ≈ $(round(tail / R0; sigdigits=2)) of |R(0)| " *
              "(tol $tol), so the table is off by about that near ρ = 0. A larger `kmax` (with a remainder written " *
              "without cancellation at large k) reduces it."
    npan = (n = ρc / scale; abs(n - round(n)) < 1e-9 ? round(Int, n) : ceil(Int, n))
    if tensor
        s = [x * (Δk[1] + Δk[2]) / 2 for (x, Δk) in zip(k, v)]; d = [x * (Δk[2] - Δk[1]) / 2 for (x, Δk) in zip(k, v)]
        table = RadialTable{2}(ρ -> _rh_direct(k, w, s, d, ρ), ρc, npan, nq)
        outs = (:α, :β)
    else
        outs = outputs === nothing ? (:value, :gradient) : Tuple(outputs)
        all(o -> o in (:value, :gradient, :potential_gradient, :potential), outs) ||
            throw(ArgumentError("SymbolKernel: unknown output in $outs"))
        regularize && outs != (:gradient,) && throw(ArgumentError("SymbolKernel: `regularize` gives only :gradient"))
        f = k .* v
        table = RadialTable{length(outs)}(ρ -> _scalar_direct(outs, k, w, f, ρ, regularize), ρc, npan, nq)
    end
    SymbolKernel(symbol, base, Float64(scale), tensor, outs, table, far, decay)
end

"""
The decay of the remainder past the grid's end `kend`: `(decay, m, tail, R0)`, with g = |k Δ| (a tensor: its larger
mobility combination) sampled on [0.45, 0.55] kend and [0.9, 1] kend, m the exponent of g ~ k^(−m) between them,
`tail` ≈ (1/2π) ∫_kend^∞ g dk (for m > 1) and R0 = (1/2π) |∫ k Δ dk| on the grid (the size of R(0)). `:exponential`
when g at kend is below 1e-14 of its maximum on the grid or decays faster than k⁻⁶, else `:algebraic`.
"""
function _tail_decay(Δ, k, w, v, kend, tensor)
    g(x, d) = tensor ? x * max(abs(d[1] + d[2]), abs(d[2] - d[1])) / 2 : x * abs(d)
    gmax = maximum(g(x, d) for (x, d) in zip(k, v))
    f = tensor ? [x * (d[1] + d[2]) / 2 for (x, d) in zip(k, v)] : k .* v
    R0 = max(abs(sum(w .* f)), 1e-300) / 2π
    win(a, b) = maximum(x -> g(x, Δ(x)), range(a * kend, b * kend; length=6))
    g1, g2 = win(0.45, 0.55), win(0.9, 1.0)
    (gmax == 0 || g2 <= 1e-14gmax) && return (:exponential, Inf, 0.0, R0)
    m = log(g1 / g2) / log(1.9)
    m > 6 && return (:exponential, m, 0.0, R0)
    (:algebraic, m, m > 1 ? g2 * kend / (m - 1) / 2π : Inf, R0)
end

# The Hankel integrands at small k, by output: k Δ (value, potential gradient, potential: J0, J1(x)/x, (1 − J0)/k² tend
# to constants), k³ Δ (gradient: k² J1(x)/x), k⁵ Δ (regularized gradient: k² (x/2 − J1)/x ~ k⁴ρ²/16); a tensor's
# A0 = ∫ s J0 k dk and A2 = ∫ d J2 k dk (J2 ~ x²/8): k s and k³ d.
function _small_k_integrands(Δk, k, tensor, outs, regularize)
    tensor && return (k * abs(Δk[1] + Δk[2]) / 2, k^3 * abs(Δk[2] - Δk[1]) / 2)
    a = abs(Δk)
    regularize && return (k^5 * a,)
    any(o -> o !== :gradient, outs) ? (k * a, k^3 * a) : (k^3 * a,)
end

"""
Throw if the remainder's Hankel integrals diverge at k = 0: an integrand that grows like 1/k (or faster) between
k = 1e-6/scale and 1e-8/scale and is not negligible there (above 1e-6 of its largest value on the k grid; values far
below are the roundoff of Δ = symbol − base symbol). (A base that matches the symbol at large k
only, as a 2D Stokeslet under a Saffman–Delbrück symbol, leaves Δ ~ k⁻² there; the table then builds, but holds a
constant fixed by the k grid.) Values that are not finite at those k leave the check undecided.
"""
function _check_small_k(Δ, k, v, scale, tensor, outs, regularize)
    k1, k2 = 1e-6 / scale, 1e-8 / scale
    Δ1, Δ2 = try
        (Δ(k1), Δ(k2))
    catch
        return nothing
    end
    f1 = _small_k_integrands(Δ1, k1, tensor, outs, regularize); f2 = _small_k_integrands(Δ2, k2, tensor, outs, regularize)
    F = [maximum(_small_k_integrands(vi, ki, tensor, outs, regularize)[n] for (vi, ki) in zip(v, k)) for n in eachindex(f1)]
    for n in eachindex(f1)
        all(isfinite, (f1[n], f2[n])) && f1[n] > 0 || continue
        growth = log(f2[n] / f1[n]) / log(k1 / k2)          # f ~ k^(−growth) as k → 0; integrable when < 1
        growth > 0.9 && f2[n] > 1e-6 * F[n] &&
            throw(ArgumentError("SymbolKernel: the remainder is not integrable at k = 0 (its Hankel integrand grows " *
                                "like k^(−$(round(growth; digits=2))) there): the base must match the symbol at small " *
                                "k as well as at large k (or, for a scalar gradient, use `regularize`)"))
    end
    nothing
end

_symbol_minus(a::Tuple, b::Tuple) = (a[1] - b[1], a[2] - b[2])
_symbol_minus(a, b) = a - b

"""The remainder's outputs at distance ρ: (α, β) of a tensor kernel, the chosen `outputs` of a scalar one."""
@inline function kernel_remainder(K::SymbolKernel, ρ)
    ρ < K.table.ρc && return K.table(ρ)
    K.far === nothing && throw(DomainError(ρ, "SymbolKernel: beyond the table (ρc = $(K.table.ρc)) and no far field"))
    K.far(ρ)
end

"""Remainder (R11, R12, R22) of a tensor `SymbolKernel` at r."""
@inline function remainder_tensor(K::SymbolKernel, r)
    α, β = kernel_remainder(K, norm(r)); (α + β * r[1]^2, β * r[1] * r[2], α + β * r[2]^2)
end

@inline function (K::SymbolKernel)(r::Vec2)
    if K.tensor
        R11, R12, R22 = remainder_tensor(K, r)
        return Mat2(K.base(r)) + Mat2(R11, R12, R12, R22)
    end
    i = findfirst(==(:value), K.outputs)
    i === nothing && throw(ArgumentError("SymbolKernel: the scalar kernel needs the output :value"))
    K.base(r) + kernel_remainder(K, norm(r))[i]
end

"""Table range ρc of a `SymbolKernel`."""
table_range(K::SymbolKernel) = K.table.ρc

"""∫ K dA over the plane = ŝ(0), the symbol's limit at k → 0 (finite when the whole kernel decays)."""
kernel_integral(K::SymbolKernel) = K.symbol(1e-12 / K.scale)

const _SYMBOL_KERNELS = Dict{Any,Any}()
const _SYMBOL_LOCK = ReentrantLock()

"""
    cached_kernel(build, key)

The kernel stored under `key` (e.g. `(:depth, H)`), built by `build()` on first use; thread-safe.
"""
cached_kernel(build, key) = lock(_SYMBOL_LOCK) do
    get!(build, _SYMBOL_KERNELS, key)
end
