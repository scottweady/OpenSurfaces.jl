# The common evaluator of a potential: S_K[f](x) = ∫ K(x − y) f(y) at a set
# of targets, from sources on an area mesh (`AreaDisc`) or on its boundary (`BoundaryDisc`), as
#   E(f) = far(f) + C f,
# the smooth rule summed by a fast method chosen by the kernel's type, plus sparse local corrections
# (`area_correction`, `line_correction`). It holds the four pieces every potential assembled by hand:
#   1. the fast sum: Ewald (or FMM3D) for `Stokeslet3D` and `LaplaceKernel3D`, FMM2D (or Ewald) for the 2D Laplace and
#      biharmonic kernels and (at wavenumber ik) for the modified Helmholtz ones, Ewald for `StokesKernel` (G2 + G3)
#      and for a `SymbolKernel` of base Stokeslet3D or LaplaceKernel3D (the whole kernel), direct sums for any other
#      callable;
#   2. the local corrections, built once;
#   3. the bookkeeping: target sets (:area, :boundary, or points; several at once share one fast sum), the unweighted
#      copy of the mesh for smooth densities given as physical values (`density = :smooth`; `:stored` takes the
#      mesh's own, edge-weighted, values), and the slots of the 2×2 kernel form that the potential uses;
#   4. a `remainder` (`SymbolKernel`) added to its base kernel: with a far field, in the fast sum (Ewald's long-range
#      multiplier); without one, the whole kernel (base + remainder, which decays) is summed directly over the pairs
#      closer than ρc, checked to lie inside the table.

"""
    kernel_shape(K) -> (nout, nin)

The slots of a kernel's 2×2 form that its potential uses: (1, 1) a scalar kernel (slot [1, 1]; the default), (2, 1) a
vector kernel (the first column), (2, 2) a tensor kernel. A kernel of your own that is not scalar declares it,
`OpenSurfaces.kernel_shape(::MyKernel) = (2, 2)`; `Evaluator` probes the kernel once and throws if a kernel taken as
scalar returns a 2×2 form with entries outside slot [1, 1] (a plain closure cannot declare a shape: wrap it in a
type).
"""
kernel_shape(::Stokeslet3D) = (2, 2)
kernel_shape(::StokesKernel) = (2, 2)
kernel_shape(::TractionKernel) = (2, 2)
kernel_shape(::LaplaceGradient2D) = (2, 1)
kernel_shape(::BiharmonicGradient2D) = (2, 1)
kernel_shape(K::SymbolKernel) = K.tensor ? (2, 2) : (1, 1)
kernel_shape(K) = (1, 1)                 # scalar kernels, in slot [1, 1]

"""A scalar kernel g(r) (or g(i, r)) as g · I, the 2×2 form of the near corrections."""
struct _ScalarAsMat{K}
    k::K
end
@inline (W::_ScalarAsMat)(r::Vec2) = (g = W.k(r); Mat2(g, 0, 0, g))
@inline (W::_ScalarAsMat)(i::Int, r::Vec2) = (g = hasmethod(W.k, Tuple{Int,Vec2}) ? W.k(i, r) : W.k(r); Mat2(g, 0, 0, g))
_log_split(K::_ScalarAsMat, r, τ) = _log_split(K.k, r, τ)

"""
Probe the kernel at two separations of the source points (target 1 for a K(i, r)): it returns a scalar (shape (1, 1)
only) or a 2×2 form; a (1, 1) kernel's 2×2 form may hold only slot [1, 1] (and [2, 2] equal to it or zero, as
g · I or diag(g, 0)). Returns whether the kernel is scalar.
"""
function _probe_kernel(K, sh, pts)
    n = length(pts); scalar = false
    tdep = hasmethod(K, Tuple{Int,Vec2})
    for r in (n >= 2 ? (pts[1] - pts[cld(n, 2)], pts[cld(n, 3)] - pts[n]) : (Vec2(0.31, 0.27),))
        r[1] == 0 && r[2] == 0 && continue
        v = tdep ? K(1, r) : K(r)
        if v isa Number
            sh == (1, 1) || throw(ArgumentError("Evaluator: the kernel $(typeof(K)) returns a scalar, but its " *
                                                "kernel_shape is $sh"))
            scalar = true; continue
        end
        size(v) == (2, 2) || throw(ArgumentError("Evaluator: a kernel returns a scalar or a 2×2 form (got " *
                                                 "$(typeof(v)) from $(typeof(K)))"))
        sh == (1, 1) || continue
        m = maximum(abs, v); tolm = 1e-14 * m
        ok = abs(v[1, 2]) <= tolm && abs(v[2, 1]) <= tolm && (abs(v[2, 2]) <= tolm || abs(v[2, 2] - v[1, 1]) <= tolm)
        ok || throw(ArgumentError("Evaluator: the kernel $(typeof(K)) returns a 2×2 form with entries outside slot " *
                                  "[1, 1], but its kernel_shape is (1, 1), so the potential would keep the xx " *
                                  "component only. Declare `OpenSurfaces.kernel_shape(::YourKernel) = (2, 2)` (or " *
                                  "(2, 1) for a vector kernel in the first column); a closure: wrap it in a type"))
    end
    scalar
end

"""3D Laplace kernel as 1/(4πρ) · I (the form of the near corrections)."""
struct _Laplace3DMat end
@inline (::_Laplace3DMat)(r::Vec2) = (1 / (4π * norm(r))) * Mat2(1, 0, 0, 1)

# A remainder that decays algebraically has a cone or a log at ρ = 0, which the smooth rule does not integrate to tol
# (Debye: 4e-5 whatever tol): the near corrections then take the whole kernel, base + remainder (`_local_sum` drops
# the self term). One that decays exponentially is smooth: the corrections of the base alone, as before.
_whole_corrections(K, R) = R isa SymbolKernel && R.decay === :algebraic && K isa Union{LaplaceKernel3D,BiharmonicGradient2D}

"""Base + remainder in the 2×2 form of the corrections: (1/(4πρ) + ℓ(ρ)) I, or ((2 log ρ + c0)/8π + q(ρ)) r."""
struct _WholeKernel{K,R}
    base::K
    R::R
end
@inline function (W::_WholeKernel{LaplaceKernel3D})(r::Vec2)
    ρ = norm(r); g = 1 / (4π * ρ) + W.R.table(ρ)[1]
    Mat2(g, 0, 0, g)
end
@inline function (W::_WholeKernel{BiharmonicGradient2D})(r::Vec2)
    ρ = norm(r); f = ((ρ == 0 ? 0.0 : 2log(ρ)) + W.base.c0) / 8π + W.R.table(ρ)[1]
    Mat2(r[1] * f, r[2] * f, 0, 0)
end

_correction_kernel(K::Stokeslet3D, scalar) = StokesKernel(Inf, K.μ)
_correction_kernel(::LaplaceKernel3D, scalar) = _Laplace3DMat()
_correction_kernel(K, scalar) = scalar ? _ScalarAsMat(K) : K      # kernels that return their 2×2 form, or g · I

_slots(C, (nout, nin)) = nout == 1 ? C[1:2:end, 1:2:end] : nin == 1 ? C[:, 1:2:end] : C

# fast sums (one per kernel family); `_far_apply(F, E, q, eps)` returns one output vector per target set
struct _EwaldFar
    plan::EwaldStokes3D
end
struct _FMM3DStokesFar
    self::Bool
end
struct _FMM3DLaplaceFar end
struct _EwaldLaplaceFar
    plan::EwaldLaplace3D                 # targets: all the sets, concatenated
end
struct _RadialEwaldFar
    plan::RadialEwald2D                  # targets: all the sets, concatenated
    kind::Symbol                         # :value, :gradient, :normal, :bgrad
end
struct _RFMM2DFar
    kind::Symbol                         # :value, :gradient, :bgrad
end
struct _HFMM2DFar
    kind::Symbol                         # :value, :normal
end
struct _DirectFar end
struct _LocalFar
    ρc::Float64
    whole::Bool                          # corrections of base + remainder (an algebraic remainder): no self term
end

"""
    Evaluator(K, ad::AreaDisc; targets=:area, bd=nothing, density=:stored, fast=:auto, remainder=nothing,
              plan=nothing, tol=DEFAULT_TOL, eps=tol, near_factor=nothing, opts=nothing, upsampling=nothing)
    Evaluator(K, bd::BoundaryDisc; targets, fast=:auto, tol=DEFAULT_TOL, eps=tol)

The potential of kernel `K` from area (or boundary) sources at `targets`: `:area` (the area nodes), `:boundary` (the
nodes of `bd`), a vector of points, or a tuple of these (one fast sum, an output per set). `E(f)` gives the potential
for the density `f` at the source nodes (scalars or `Vec2`, by `kernel_shape(K)`); `evaluator_matrix(E)` the dense
matrix of a direct-sum evaluator.
- `density`: `:stored` takes the mesh's own values (edge-weighted on its weighted panels); `:smooth` physical nodal
  values of a smooth density, integrated on the unweighted copy of the mesh.
- `fast`: `:auto` (by the kernel's type: Ewald for `Stokeslet3D`, `LaplaceKernel3D`, `StokesKernel` and a
  `SymbolKernel` of either 3D base, FMM2D for the 2D kernels, direct sums for other callables), `:ewald`, `:fmm`,
  `:direct`.
- `remainder`: a `SymbolKernel` added to `K` (head of src/evaluators/evaluator.jl); for `Stokeslet3D` at
  μ = 1 (scaled with `K`), into Ewald's multiplier; for `LaplaceKernel3D` and `BiharmonicGradient2D` without a far
  field, a local sum over the pairs closer than ρc (where the kernel's tail falls below `eps`).
- `plan`: an existing `EwaldStokes3D` (`EwaldLaplace3D`: targets = all the sets, concatenated) plan of the same
  sources and targets to reuse.
- `near_factor`, `opts`, `upsampling`: passed to `area_correction`. With an `AreaUpsampling` of `ad` (stored
  densities, no remainder) the fast sum runs over its points, the density interpolated to them (`upsample`).
- `tol`: the local corrections; `eps`: the fast sum (fixed here for Ewald and the local sum of a remainder; per call
  `E(f; eps)` for the FMMs).

As a linear operator (one target set) on interleaved `Float64` vectors (plain ones for a scalar kernel):
`size(E)`, `mul!(y, E, x)`, `E * x`; `evaluator_matrix(E)` is its dense matrix and `local_blocks(E, groups)` the
diagonal blocks of panel groups (`panel_groups`).
"""
struct Evaluator{K,R,F}
    kernel::K
    remainder::R
    shape::NTuple{2,Int}
    src::Vector{Vec2}
    w::Vector{Float64}
    smooth::Union{Nothing,Tuple{AreaDisc,Matrix{Float64}}}    # :smooth: the mesh and its Gauss–Jacobi → Legendre map
    up::Union{Nothing,Tuple{AreaUpsampling,AreaDisc}}          # the far sum on an upsampled rule of the mesh
    ad::Union{Nothing,AreaDisc}                                # the source mesh (area sources)
    trg::Vector{Vector{Vec2}}
    C::Vector{SparseMatrixCSC{Float64,Int}}
    far::F
    eps::Float64
end

function Evaluator(K, ad::AreaDisc; targets=:area, bd=nothing, density::Symbol=:stored, fast::Symbol=:auto,
                   remainder=nothing, plan=nothing, tol::Real=DEFAULT_TOL, eps::Real=tol, near_factor=nothing,
                   opts=nothing, upsampling=nothing)
    density in (:stored, :smooth) || throw(ArgumentError("Evaluator: density is :stored or :smooth"))
    if upsampling !== nothing
        density === :stored && remainder === nothing ||
            throw(ArgumentError("Evaluator: upsampling needs density = :stored and no remainder"))
        length(upsampling.x) == npanels(ad) * upsampling.q^2 ||
            throw(ArgumentError("Evaluator: the AreaUpsampling is not of this mesh"))
    end
    ckw = (; tol, near_factor, opts, upsampling)
    sets = targets isa Tuple ? collect(targets) : [targets]
    if density === :smooth
        srcd, Iw, ainfo = _unweighted_copy(ad); smooth = (ad, Iw)
    else
        srcd = ad; ainfo = nothing; smooth = nothing
    end
    trg = Vector{Vec2}[]; C = SparseMatrixCSC{Float64,Int}[]
    sh = kernel_shape(K); Kc = _correction_kernel(K, _probe_kernel(K, sh, ad.x))
    whole = _whole_corrections(K, remainder)
    whole && (Kc = _WholeKernel(K, remainder))
    for t in sets
        if t === :area
            push!(trg, ad.x)
            push!(C, density === :stored ? area_correction(srcd, Kc; ckw...) :
                                           area_correction(srcd, Kc; targets=ad.x, target_info=ainfo, ckw...))
        elseif t === :boundary
            bd === nothing && throw(ArgumentError("Evaluator: targets = :boundary needs `bd`"))
            push!(trg, bd.x)
            push!(C, area_correction(srcd, Kc; targets=bd.x, target_info=boundary_target_info(srcd, bd), ckw...))
        else
            X = collect(Vec2, t); push!(trg, X)
            push!(C, area_correction(srcd, Kc; targets=X, ckw...))
        end
    end
    C = [_slots(c, sh) for c in C]
    src, w = upsampling === nothing ? (srcd.x, srcd.w) : (upsampling.x, upsampling.w)
    self = density === :stored && sets == [:area] && upsampling === nothing
    far = _area_far(K, remainder, fast, plan, src, trg, self, C, eps)
    Evaluator(K, remainder, sh, src, w, smooth, upsampling === nothing ? nothing : (upsampling, ad), ad, trg, C, far,
              Float64(eps))
end

function Evaluator(K, bd::BoundaryDisc; targets, fast::Symbol=:auto, tol::Real=DEFAULT_TOL, eps::Real=tol)
    sets = targets isa Tuple ? collect(targets) : [targets]
    trg = [collect(Vec2, t) for t in sets]
    lq = LineQuad(bd; tol=tol); sh = kernel_shape(K)
    Kc = _correction_kernel(K, _probe_kernel(K, sh, bd.x))
    C = [_slots(line_correction(Kc, lq, X), sh) for X in trg]
    far = fast === :direct || !(K isa MHKernel) ? _DirectFar() : _HFMM2DFar(:value)
    Evaluator(K, nothing, sh, bd.x, bd.w, nothing, nothing, nothing, trg, C, far, Float64(eps))
end

# the fast sum of area sources, by kernel
function _area_far(K::Stokeslet3D, R, fast, plan, src, trg, self, C, eps)
    length(trg) == 1 || throw(ArgumentError("Evaluator: one target set for Stokeslet3D"))
    if fast === :direct                  # with a remainder: the direct sum of G3 + R/μ (`_far_apply(::_DirectFar)`)
        R === nothing || R.decay !== :algebraic ||
            throw(ArgumentError("Evaluator: Stokeslet3D's corrections are the Stokeslet's, which an algebraically " *
                                "decaying remainder needs too; use the SymbolKernel itself as the kernel"))
        return _DirectFar()
    end
    fast === :fmm && (R === nothing || throw(ArgumentError("Evaluator: a remainder needs fast = :ewald"));
                      return _FMM3DStokesFar(self))
    plan !== nothing && return _EwaldFar(plan)
    _EwaldFar(self ? EwaldStokes3D(src; μ_3D=K.μ, remainder=R, tol=eps) :
                     EwaldStokes3D(src, trg[1]; μ_3D=K.μ, remainder=R, tol=eps))
end
function _area_far(K::Union{LaplaceKernel3D,BiharmonicGradient2D}, R, fast, plan, src, trg, self, C, eps)
    if R !== nothing
        R.far === nothing || throw(ArgumentError("Evaluator: a remainder with a far field is summed in Ewald's " *
                                                 "multiplier (Stokeslet3D) only"))
        sc = K isa LaplaceKernel3D
        dmax = maximum(_max_pair_distance(c, X, src; interleaved=!sc) for (c, X) in zip(C, trg))
        dmax < table_range(R) ||
            error("Evaluator: near corrections reach $(dmax / R.scale) × scale, beyond the remainder's table " *
                  "($(table_range(R) / R.scale) × scale)")
        return _LocalFar(max(sc ? _tail_cutoff(R, eps) : _gradient_cutoff(R, K.c0, eps), 1.01dmax),
                         R.decay === :algebraic)
    end
    fast === :direct && return _DirectFar()
    K isa BiharmonicGradient2D && return fast === :ewald ? _radial_far(:log, :bgrad, src, trg, 0.0, eps) : _RFMM2DFar(:bgrad)
    fast in (:auto, :ewald, :fmm) || throw(ArgumentError("Evaluator: fast = :auto, :ewald, :fmm or :direct for LaplaceKernel3D"))
    fast === :fmm && plan === nothing && return _FMM3DLaplaceFar()
    plan === nothing || plan isa EwaldLaplace3D ||
        throw(ArgumentError("Evaluator: the plan of a LaplaceKernel3D is an EwaldLaplace3D"))
    plan !== nothing && return _EwaldLaplaceFar(plan)
    X = reduce(vcat, trg)
    _EwaldLaplaceFar(length(trg) == 1 && X == src ? EwaldLaplace3D(src; tol=eps) : EwaldLaplace3D(src, X; tol=eps))
end
function _area_far(K::Union{LaplaceKernel2D,LaplaceGradient2D,MHKernel,MHNormalKernel}, R, fast, plan, src, trg, self, C, eps)
    R === nothing || throw(ArgumentError("Evaluator: no remainder for $(typeof(K))"))
    fast === :direct && return _DirectFar()
    fast in (:auto, :ewald, :fmm) || throw(ArgumentError("Evaluator: fast = :auto, :ewald, :fmm or :direct for $(typeof(K))"))
    if fast === :ewald                   # :auto: FMM2D, faster than Ewald at the same accuracy (studies/laplace_ewald)
        return K isa LaplaceKernel2D ? _radial_far(:log, :value, src, trg, 0.0, eps) :
               K isa LaplaceGradient2D ? _radial_far(:log, :gradient, src, trg, 0.0, eps) :
               K isa MHKernel ? _radial_far(:mh, :value, src, trg, K.k, eps) : _radial_far(:mh, :normal, src, trg, K.k, eps)
    end
    K isa LaplaceKernel2D ? _RFMM2DFar(:value) : K isa LaplaceGradient2D ? _RFMM2DFar(:gradient) :
        K isa MHKernel ? _HFMM2DFar(:value) : _HFMM2DFar(:normal)
end
function _radial_far(kind, out, src, trg, k, eps)
    X = reduce(vcat, trg)
    kw = (; k, out=out in (:gradient, :normal) ? :gradient : :value, nd=out === :bgrad ? 3 : 1, tol=eps)
    _RadialEwaldFar(length(trg) == 1 && X == src ? RadialEwald2D(kind, src; kw...) : RadialEwald2D(kind, src, X; kw...), out)
end
# a SymbolKernel as the kernel itself (base Stokeslet3D or LaplaceKernel3D): Ewald of the whole kernel, its remainder
# split by weight (any decay; src/kernels/ewald.jl), coincident pairs skipped as by the direct sum
function _area_far(K::SymbolKernel, R, fast, plan, src, trg, self, C, eps)
    R === nothing || throw(ArgumentError("Evaluator: no remainder for $(typeof(K))"))
    fast in (:auto, :ewald, :direct) ||
        throw(ArgumentError("Evaluator: fast = :auto, :ewald or :direct for a SymbolKernel"))
    ok = K.base isa Stokeslet3D ? K.tensor : K.base isa LaplaceKernel3D && !K.tensor && :value in K.outputs
    if fast === :direct || !ok
        fast === :ewald && throw(ArgumentError("Evaluator: Ewald sums a SymbolKernel with base Stokeslet3D (tensor) " *
                                               "or LaplaceKernel3D (scalar, with :value)"))
        return _DirectFar()
    end
    X = reduce(vcat, trg)
    if K.base isa Stokeslet3D
        length(trg) == 1 || throw(ArgumentError("Evaluator: one target set for a Stokes SymbolKernel under Ewald"))
        return _EwaldFar(self ? EwaldStokes3D(src; μ_3D=K.base.μ, remainder=K, whole=true, tol=eps) :
                                EwaldStokes3D(src, X; μ_3D=K.base.μ, remainder=K, whole=true, tol=eps))
    end
    _EwaldLaplaceFar(length(trg) == 1 && X == src ? EwaldLaplace3D(src; remainder=K, tol=eps) :
                                                  EwaldLaplace3D(src, X; remainder=K, tol=eps))
end
# StokesKernel(μ_2D, μ_3D) = G2 + G3: one Ewald plan of both (Hasimoto's split of G2, the erf split of G3)
function _area_far(K::StokesKernel, R, fast, plan, src, trg, self, C, eps)
    R === nothing || throw(ArgumentError("Evaluator: no remainder for StokesKernel (use Stokeslet3D with a remainder)"))
    fast in (:auto, :ewald, :direct) || throw(ArgumentError("Evaluator: fast = :auto, :ewald or :direct for StokesKernel"))
    (fast === :direct || !(isfinite(K.μ_2D) || isfinite(K.μ_3D))) && return _DirectFar()
    length(trg) == 1 || throw(ArgumentError("Evaluator: one target set for StokesKernel under Ewald"))
    _EwaldFar(self ? EwaldStokes3D(src; μ_3D=K.μ_3D, μ_2D=K.μ_2D, tol=eps) :
                     EwaldStokes3D(src, trg[1]; μ_3D=K.μ_3D, μ_2D=K.μ_2D, tol=eps))
end
function _area_far(K, R, fast, plan, src, trg, self, C, eps)
    R === nothing || throw(ArgumentError("Evaluator: no remainder for $(typeof(K))"))
    _DirectFar()
end

"""
    E(f; eps=E.eps)

The potential of `f` (values at the source nodes) at the targets: a vector, or a tuple of them (several sets). `eps`
sets the fast sum's accuracy per call for the FMM paths only; the Ewald plan and the local sum of a remainder fix
theirs at construction, so there a different `eps` is an `ArgumentError` (build another evaluator instead).
"""
function (E::Evaluator)(f::AbstractVector; eps::Real=E.eps)
    eps == E.eps || !(E.far isa Union{_EwaldFar,_EwaldLaplaceFar,_RadialEwaldFar,_LocalFar}) ||
        throw(ArgumentError("Evaluator: eps = $eps differs from the $(E.eps) this evaluator's " *
                            "$(E.far isa _LocalFar ? "local sum" : "Ewald plan") was built for; it cannot change per call"))
    f̃ = E.smooth === nothing ? f : _to_unweighted(E.smooth[1], E.smooth[2], f)
    U = _far_apply(E.far, E, (E.up === nothing ? f̃ : upsample(E.up[1], E.up[2], f̃)) .* E.w, Float64(eps))
    nout, nin = E.shape
    out = map(eachindex(E.trg)) do s
        nout == 1 ? U[s] .+ E.C[s] * f̃ : nin == 1 ? U[s] .+ deinterleave(E.C[s] * f̃) :
                    U[s] .+ deinterleave(E.C[s] * interleave(f̃))
    end
    length(out) == 1 ? out[1] : Tuple(out)
end

_split(v::AbstractVector, trg) = (o = cumsum([0; length.(trg)]); [v[o[s]+1:o[s+1]] for s in eachindex(trg)])
_split(m::AbstractMatrix, trg) = (o = cumsum([0; length.(trg)]); [m[:, o[s]+1:o[s+1]] for s in eachindex(trg)])

_far_apply(F::_EwaldFar, E, q, eps) = [ewald_apply(F.plan, q)]
_far_apply(F::_EwaldLaplaceFar, E, q, eps) = _split(ewald_apply(F.plan, q), E.trg)
function _far_apply(F::_RadialEwaldFar, E, q, eps)
    F.kind in (:value, :gradient) && return _split(ewald_apply(F.plan, q), E.trg)
    X = reduce(vcat, E.trg)
    if F.kind === :normal
        g = ewald_apply(F.plan, q); n = E.kernel.n
        return _split([dot(g[i], n[i]) for i in eachindex(X)], E.trg)
    end
    # :bgrad, as the FMM path: (x − y)(2 log ρ + c0) = 2(x P₀ − P_y) + c0 (x Q − Q_y), P = Σ q {1, y} log ρ
    y1 = first.(E.src); y2 = last.(E.src)
    p = 2π .* ewald_apply(F.plan, [q q .* y1 q .* y2])
    c0 = E.kernel.c0
    Q = sum(q); Qy = Vec2(sum(q .* y1), sum(q .* y2))
    _split([(2 * (X[i] * p[i, 1] - Vec2(p[i, 2], p[i, 3])) + c0 * (X[i] * Q - Qy)) / 8π for i in eachindex(X)], E.trg)
end
_far_apply(F::_FMM3DStokesFar, E, q, eps) =
    [F.self ? stokes3d_plane_apply(E.src, q; μ_3D=E.kernel.μ, eps=eps).u :
              stokes3d_plane_apply(E.src, q, E.trg[1]; μ_3D=E.kernel.μ, eps=eps).u]

function _far_apply(::_FMM3DLaplaceFar, E, q, eps)
    X = reduce(vcat, E.trg)
    src = zeros(3, length(E.src)); trg = zeros(3, length(X))
    for (i, x) in enumerate(E.src); src[1, i] = x[1]; src[2, i] = x[2]; end
    for (i, x) in enumerate(X); trg[1, i] = x[1]; trg[2, i] = x[2]; end
    v = FMM3D.lfmm3d(eps, src; charges=q, targets=trg, pgt=1)
    v.ier == 0 || error("lfmm3d returned ier = $(v.ier)")
    _split(v.pottarg, E.trg)                                  # FMM3D Laplace returns Σ q/(4π|r|)
end

function _far_apply(F::_RFMM2DFar, E, q, eps)
    X = reduce(vcat, E.trg)
    if F.kind === :bgrad
        # (x − y)(2 log ρ + c0) summed as 2(x P₀ − P_y) + c0 (x Q − Q_y), P = Σ q {1, y} log ρ
        y1 = first.(E.src); y2 = last.(E.src)
        v = FMM2D.rfmm2d(eps=eps, sources=_pts(E.src), charges=Matrix([q q .* y1 q .* y2]'), targets=_pts(X), pgt=1, nd=3)
        v.ier == 0 || error("rfmm2d returned ier = $(v.ier)")
        c0 = E.kernel.c0
        Q = sum(q); Qy = Vec2(sum(q .* y1), sum(q .* y2))
        far(x, P) = (2 * (x * P[1] - Vec2(P[2], P[3])) + c0 * (x * Q - Qy)) / 8π
        p = real.(v.pottarg)
        return _split([far(X[i], view(p, :, i)) for i in eachindex(X)], E.trg)
    end
    v = FMM2D.rfmm2d(eps=eps, sources=_pts(E.src), charges=q, targets=_pts(X), pgt=2)
    v.ier == 0 || error("rfmm2d returned ier = $(v.ier)")
    F.kind === :value ? _split(real.(v.pottarg) ./ 2π, E.trg) : _split(_vecs(real.(v.gradtarg) ./ 2π), E.trg)
end

function _far_apply(F::_HFMM2DFar, E, q, eps)
    X = reduce(vcat, E.trg)
    v = FMM2D.hfmm2d(eps=eps, zk=complex(0.0, E.kernel.k), sources=_pts(E.src), charges=complex.(q), targets=_pts(X),
                     pgt=F.kind === :value ? 1 : 2)
    v.ier == 0 || error("hfmm2d returned ier = $(v.ier)")
    F.kind === :value && return _split(real.(v.pottarg), E.trg)
    g = real.(v.gradtarg); n = E.kernel.n
    _split([g[1, i] * n[i][1] + g[2, i] * n[i][2] for i in eachindex(X)], E.trg)
end

function _far_apply(::_DirectFar, E, q, eps)
    K = E.kernel; nout, nin = E.shape
    tdep = hasmethod(K, Tuple{Int,Vec2})
    E.remainder === nothing && return [_direct_sum(K, Val(tdep), Val(nout), Val(nin), E.src, q, X) for X in E.trg]
    # Stokeslet3D + remainder (a tensor SymbolKernel at μ = 1): as in Ewald, the remainder over all pairs, R(0) = α(0) I
    # at the coincident ones
    KR = _PlusRemainder(K.μ, E.remainder); α0 = E.remainder.table(0.0)[1] / K.μ
    map(E.trg) do X
        u = _direct_sum(KR, Val(false), Val(2), Val(2), E.src, q, X)
        for (i, j) in _coincident_pairs(E.src, X, false)
            u[i] += α0 * q[j]
        end
        u
    end
end

"""G3(r; μ) + R(r)/μ, R a tensor `SymbolKernel` remainder at μ = 1."""
struct _PlusRemainder{R}
    μ::Float64
    R::R
end
@inline (W::_PlusRemainder)(r::Vec2) = ((R11, R12, R22) = remainder_tensor(W.R, r);
                                         G3(r; μ_3D=W.μ) + Mat2(R11, R12, R12, R22) / W.μ)
# Σ_j K(x_i − y_j) q_j, coincident pairs skipped (a function barrier on the kernel and its slots)
function _direct_sum(K, ::Val{tdep}, ::Val{nout}, ::Val{nin}, src, q, X) where {tdep,nout,nin}
    out = zeros(nout == 1 ? Float64 : Vec2, length(X))
    Threads.@threads for i in eachindex(X)
        x = X[i]; acc = zero(eltype(out))
        @inbounds for j in eachindex(src)
            r = x - src[j]
            r[1] == 0 && r[2] == 0 && continue
            M = tdep ? K(i, r) : K(r)
            acc += nout == 1 ? M[1, 1] * q[j] : nin == 1 ? Vec2(M[1, 1], M[2, 1]) * q[j] : Mat2(M) * q[j]
        end
        out[i] = acc
    end
    out
end

function _far_apply(F::_LocalFar, E, q, eps)
    K = E.kernel; R = E.remainder; src = E.src; ρc = F.ρc
    cells, cell = _cell_list(src, ρc)
    map(E.trg) do T
        _local_sum(K, R, q, src, T, cells, cell, ρc, !F.whole)
    end
end

"""Σ_{|x−y| < ρc} (1/(4πρ) + ℓ(ρ)) q at the targets T; with `self`, a target on a source node takes its smooth part
ℓ(0) there (ℓ smooth, integrated by the smooth rule; without it, the corrections hold the whole kernel there)."""
function _local_sum(::LaplaceKernel3D, R::SymbolKernel, q, src, T, cells, cell, ρc, self::Bool=true)
    l0 = R.table(0.0)[1]
    out = zeros(length(T))
    _chunked_foreach(length(T)) do b, i
        _neighbors!(b, T[i], src, cells, cell, ρc^2, 0)
        acc = self && i <= length(src) && T[i] == src[i] ? l0 * q[i] : 0.0   # the self pair (dropped by _neighbors!)
        @inbounds for j in b
            ρ = norm(T[i] - src[j])
            acc += q[j] * (1 / (4π * ρ) + R.table(ρ)[1])              # table clamped: ρc ≤ 1.01 × its range
        end
        out[i] = acc
    end
    out
end
"""Σ_{|x−y| < ρc} ((2 log ρ + c0)/8π + q(ρ)) r q at the targets T (no self term: r = 0)."""
function _local_sum(K::BiharmonicGradient2D, R::SymbolKernel, q, src, T, cells, cell, ρc, self::Bool=true)
    out = zeros(Vec2, length(T))
    _chunked_foreach(length(T)) do b, i
        _neighbors!(b, T[i], src, cells, cell, ρc^2, 0)
        acc = zero(Vec2)
        @inbounds for j in b
            r = T[i] - src[j]; ρ = norm(r)
            acc += (q[j] * ((2log(ρ) + K.c0) / 8π + R.table(ρ)[1])) * r     # table clamped: ρc ≤ 1.01 × its range
        end
        out[i] = acc
    end
    out
end

# ---------------------------------------------------------------------------------------
# The evaluator as a matrix: dense, its blocks, and the linear-operator interface
# ---------------------------------------------------------------------------------------

"""
The pair kernel of the smooth sum, as the dense matrix sees it: `(K, self)`, with K(r) (or K(i, r)) the kernel of the
pairs at distinct points (zero beyond the cutoff of a local sum) and `self` the 2×2 value at coincident pairs
(`nothing`: none). As the apply: Stokeslet3D + remainder over all pairs, R(0) at coincident ones; a local sum of base +
remainder within ρc, ℓ(0) at coincident pairs unless the corrections hold the whole kernel.
"""
function _pair_kernel(E::Evaluator)
    F = E.far; K = E.kernel; R = E.remainder
    if F isa _LocalFar
        W = _WholeKernel(K, R)
        cut = r -> (norm(r) < F.ρc ? W(r) : zero(Mat2))
        self = K isa LaplaceKernel3D && !F.whole ? (l = R.table(0.0)[1]; Mat2(l, 0, 0, l)) : nothing
        return cut, self
    end
    R === nothing && return K, nothing
    α0 = R.table(0.0)[1] / K.μ
    _PlusRemainder(K.μ, R), Mat2(α0, 0, 0, α0)
end

_as_mat(v::Number) = Mat2(v, 0, 0, v)
_as_mat(v) = Mat2(v)

# Σ_j K(x_i − y_j) w_j in the layout of the shape (columns: the source rule's points; `ti`: the global index of each
# target, for a K(i, r)); a function barrier
function _smooth_fill!(M, K, ::Val{tdep}, ::Val{nout}, ::Val{nin}, X, src, w, self, ti=eachindex(X)) where {tdep,nout,nin}
    Threads.@threads for j in eachindex(src)
        y = src[j]; wj = w[j]
        @inbounds for i in eachindex(X)
            r = X[i] - y
            B = r[1] == 0 && r[2] == 0 ? (self === nothing ? continue : self) : _as_mat(tdep ? K(ti[i], r) : K(r))
            B = B * wj
            if nout == 1
                M[i, j] = B[1, 1]
            elseif nin == 1
                M[2i-1, j] = B[1, 1]; M[2i, j] = B[2, 1]
            else
                M[2i-1, 2j-1] = B[1, 1]; M[2i, 2j-1] = B[2, 1]; M[2i-1, 2j] = B[1, 2]; M[2i, 2j] = B[2, 2]
            end
        end
    end
    M
end

# M += C over C's nonzeros (dense + sparse without a dense copy of C)
function _add_sparse!(M::Matrix{Float64}, C::SparseMatrixCSC)
    rv = rowvals(C); nz = nonzeros(C)
    @inbounds for j in axes(C, 2), k in nzrange(C, j)
        M[rv[k], j] += nz[k]
    end
    M
end

# the map from the nodal density to the values at the source rule's points: identity (`nothing`), the unweighted
# copy's interpolation (:smooth) or the upsampling (both sparse, per panel), interleaved for vector densities
function _source_map(E::Evaluator)
    nin = E.shape[2]
    if E.smooth !== nothing
        ad, Iw = E.smooth; p = ad.p
        T = _panel_blockdiag(ad, Matrix(1.0I, p^2, p^2), kron(Iw, Matrix(1.0I, p, p)))
    elseif E.up !== nothing
        up, ad = E.up
        T = _panel_blockdiag(ad, up.Iu, up.Iw)
    else
        return nothing
    end
    nin == 1 ? T : kron(T, sparse(1.0I, 2, 2))
end

# block diagonal over the panels: `Bu` on plain panels, `Bw` on weighted ones (same size)
function _panel_blockdiag(ad::AreaDisc, Bu::AbstractMatrix, Bw::AbstractMatrix)
    m, n = size(Bu); I_ = Int[]; J_ = Int[]; V_ = Float64[]
    for (k, P) in enumerate(ad.panels)
        B = P.weighted ? Bw : Bu
        for b in 1:n, a in 1:m
            B[a, b] == 0 && continue
            push!(I_, (k - 1) * m + a); push!(J_, (k - 1) * n + b); push!(V_, B[a, b])
        end
    end
    sparse(I_, J_, V_, m * npanels(ad), n * npanels(ad))
end

"""
    evaluator_matrix(E) -> Matrix

The dense matrix of an `Evaluator` with one target set, acting on the density at the source nodes: the smooth rule
(coincident pairs skipped; a remainder over all pairs, with R(0) at coincident ones; a local sum within its cutoff),
plus the corrections. Its layout is the kernel's shape: Nt × N (scalar), 2Nt × N (vector kernel), 2Nt × 2N
(tensor), interleaved, so `M * interleave(f) ≈ interleave(E(f))` (scalar: `M * f ≈ E(f)`); for a fast sum, to its
`eps`. With `density = :smooth` or `upsampling`, the interpolation from the nodes is included.
"""
function evaluator_matrix(E::Evaluator)
    length(E.trg) == 1 || throw(ArgumentError("evaluator_matrix: one target set"))
    X = E.trg[1]; nout, nin = E.shape
    K, self = _pair_kernel(E); tdep = hasmethod(K, Tuple{Int,Vec2})
    S = _smooth_fill!(zeros(nout * length(X), nin * length(E.src)), K, Val(tdep), Val(nout), Val(nin), X, E.src, E.w, self)
    T = _source_map(E)
    T === nothing && return _add_sparse!(S, E.C[1])
    E.up === nothing ? _add_sparse!(S, E.C[1]) * T : _add_sparse!(S * T, E.C[1])
end

"""
    panel_groups(ad; group=true) -> Vector{Vector{Int}}

Groups of area panels for block preconditioners: each edge column (the panels over one boundary panel: the weighted
layer and the graded layers above it, thin and strongly coupled) and each interior panel alone; `group = false`: one
group per panel.
"""
function panel_groups(ad::AreaDisc; group::Bool=true)
    group || return [[k] for k in 1:npanels(ad)]
    cols = Dict{Int,Vector{Int}}(); out = Vector{Vector{Int}}()
    for (k, P) in enumerate(ad.panels)
        if P.bpanel > 0
            push!(get!(cols, P.bpanel, Int[]), k)
        else
            push!(out, [k])
        end
    end
    [[cols[b] for b in sort!(collect(keys(cols)))]; out]
end

"""
    local_blocks(E, groups) -> (idx, blocks)

The diagonal blocks of `evaluator_matrix(E)` on groups of area panels (e.g. `panel_groups(ad)`): for each group, the
indices `idx[g]` of its unknowns in the evaluator's layout (interleaved for a tensor kernel) and the dense block
`blocks[g]` (the smooth rule within the group plus the corrections, from the nodal density). For an evaluator whose
targets are its source mesh's nodes (`targets = :area`, a scalar or tensor kernel, no upsampling); e.g. a
block-Jacobi preconditioner applies `blocks[g] \\ y[idx[g]]` on each group.
"""
function local_blocks(E::Evaluator, groups)
    ok = E.ad !== nothing && length(E.trg) == 1 && E.trg[1] === E.ad.x && E.up === nothing && E.shape[1] == E.shape[2]
    ok || throw(ArgumentError("local_blocks: an evaluator of area sources at their own nodes (targets = :area), a " *
                              "scalar or tensor kernel, no upsampling"))
    ad = E.ad; p2 = ad.p^2; X = E.trg[1]; d = E.shape[1]
    K, self = _pair_kernel(E); tdep = hasmethod(K, Tuple{Int,Vec2})
    T = _source_map(E)
    idx = Vector{Vector{Int}}(undef, length(groups)); blocks = Vector{Matrix{Float64}}(undef, length(groups))
    Threads.@threads :dynamic for g in eachindex(groups)
        nodes = reduce(vcat, [collect((k - 1) * p2 + 1:k * p2) for k in groups[g]])
        I_ = d == 1 ? nodes : reduce(vcat, [[2i - 1, 2i] for i in nodes])
        B = zeros(length(I_), length(I_))
        _smooth_fill!(B, K, Val(tdep), Val(d), Val(d), X[nodes], E.src[nodes], E.w[nodes], self, nodes)
        _add_sparse!(B, E.C[1][I_, I_])
        idx[g] = I_; blocks[g] = T === nothing ? B : B * Matrix(T[I_, I_])
    end
    idx, blocks
end

_ndensity(E::Evaluator) = E.up === nothing ? length(E.src) : length(E.up[2].x)

function Base.size(E::Evaluator)
    length(E.trg) == 1 || throw(ArgumentError("an Evaluator with several target sets has no single size"))
    (E.shape[1] * length(E.trg[1]), E.shape[2] * _ndensity(E))
end
Base.size(E::Evaluator, d::Integer) = size(E)[d]
Base.eltype(::Evaluator) = Float64

# the apply on a flat vector: interleaved for vector densities and outputs
function _flat_apply(E::Evaluator, x::AbstractVector{<:Real})
    length(x) == size(E, 2) || throw(DimensionMismatch("Evaluator of size $(size(E)) applied to a vector of length $(length(x))"))
    f = E.shape[2] == 1 ? x : [Vec2(x[2j-1], x[2j]) for j in 1:length(x) ÷ 2]
    u = E(f)
    E.shape[1] == 1 ? u : interleave(u)
end

"""
    mul!(y, E::Evaluator, x[, α, β])

y = E x (or α E x + β y) on flat `Float64` vectors: interleaved (x₁, y₁, x₂, y₂, …) for vector densities and
outputs, plain for scalar ones, so that an `Evaluator` is a linear operator for Krylov solvers.
"""
function LinearAlgebra.mul!(y::AbstractVector, E::Evaluator, x::AbstractVector)
    length(y) == size(E, 1) || throw(DimensionMismatch("output of length $(length(y)), Evaluator of size $(size(E))"))
    copyto!(y, _flat_apply(E, x))
end
function LinearAlgebra.mul!(y::AbstractVector, E::Evaluator, x::AbstractVector, α::Number, β::Number)
    length(y) == size(E, 1) || throw(DimensionMismatch("output of length $(length(y)), Evaluator of size $(size(E))"))
    u = _flat_apply(E, x)
    iszero(β) ? (y .= α .* u) : (y .= α .* u .+ β .* y)
end
Base.:*(E::Evaluator, x::AbstractVector{<:Real}) = mul!(zeros(size(E, 1)), E, x)

# ---------------------------------------------------------------------------------------
# Helpers: the unweighted copy of a mesh and smooth densities on it, point layouts, decay radii
# ---------------------------------------------------------------------------------------

"""The film mesh with every panel unweighted (sources for smooth densities), the Gauss–Jacobi
→ Gauss–Legendre interpolation matrix in t, and `target_info` placing the film nodes on it."""
function _unweighted_copy(ad::AreaDisc)
    ads = AreaDisc([Panel(P.B, P.T, false, P.bpanel) for P in ad.panels], ad.p; pstart=ad.pstart)
    p2 = ad.p^2
    ainfo = [((i - 1) ÷ p2 + 1, (ad.st[i][1], ad.st[i][2])) for i in eachindex(ad.x)]
    (ads, interp_matrix(ad.rtw, ad.rt.x), ainfo)
end

"""Values of a smooth density at the unweighted copy's nodes, from physical values at the film nodes."""
function _to_unweighted(ad::AreaDisc, Iw::AbstractMatrix, g::AbstractVector{<:Real})
    p = ad.p; p2 = p^2
    out = collect(Float64, g)
    for (k, P) in enumerate(ad.panels)
        P.weighted || continue
        r = (k - 1) * p2 .+ (1:p2)
        out[r] = vec(reshape(g[r], p, p) * transpose(Iw))
    end
    out
end
# vector densities: each component
_to_unweighted(ad::AreaDisc, Iw::AbstractMatrix, g::AbstractVector{<:StaticVector{2}}) =
    Vec2.(_to_unweighted(ad, Iw, [v[1] for v in g]), _to_unweighted(ad, Iw, [v[2] for v in g]))

_pts(x::AbstractVector{Vec2}) = (m = zeros(2, length(x)); for (i, v) in enumerate(x); m[1, i] = v[1]; m[2, i] = v[2]; end; m)
_vecs(m::AbstractMatrix) = [Vec2(m[1, i], m[2, i]) for i in axes(m, 2)]

"""Largest target–source distance among the nonzeros of a correction matrix (rows interleaved by target, or one
row per target with `interleaved=false`)."""
function _max_pair_distance(C::SparseMatrixCSC, T, src; interleaved::Bool=true)
    d = 0.0; rv = rowvals(C)
    for j in axes(C, 2), k in nzrange(C, j)
        d = max(d, norm(T[interleaved ? (rv[k] + 1) ÷ 2 : rv[k]] - src[j]))
    end
    d
end

"""Distance beyond which ρ|∇K| (the bilaplacian gradient (2 log ρ + c0)/8π plus the remainder's q) stays below eps of
its maximum (eps raised to the table's floor): 3000 samples on the table range."""
function _gradient_cutoff(K::SymbolKernel, c0, eps)
    ρs = collect(range(0, table_range(K); length=3001))[2:end]
    g = [ρ * abs((2log(ρ) + c0) / 8π + K.table(ρ)[1]) for ρ in ρs]
    env = reverse(accumulate(max, reverse(g))) ./ maximum(g)
    ρs[findfirst(<=(max(eps, 2minimum(env))), env)]
end

"""Distance beyond which the kernel base + remainder carries less than eps of its integral ∫ K dA (eps raised to the
table's floor): 6000 samples on the table range."""
function _tail_cutoff(K::SymbolKernel, eps)
    ρs = collect(range(0, table_range(K); length=6001))[2:end]; dρ = ρs[2] - ρs[1]
    f = [2π * ρ * abs(K.base(Vec2(ρ, 0)) + K.table(ρ)[1]) for ρ in ρs]       # |K| times the ring length
    tail = reverse(cumsum(reverse(f))) .* dρ ./ kernel_integral(K)
    ρs[findfirst(<=(max(eps, 2minimum(tail))), tail)]
end
