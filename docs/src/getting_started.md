# Getting started

## Installation

OpenSurfaces is not registered (Julia 1.12 or later; dependencies FFTW, FINUFFT, FMM2D, FMM3D, Gmsh,
SpecialFunctions, StaticArrays, and the standard libraries LinearAlgebra and SparseArrays). Add it by URL, or develop
a local copy by path:

```julia
using Pkg
Pkg.add(url = "https://github.com/scottweady/OpenSurfaces.jl")
# or: Pkg.develop(path = "path/to/OpenSurfaces")
```

The fast sums and the quadrature are threaded: start Julia with threads (`julia -t 8`). The tests run from the
package root with `julia -t 8 --project -e 'using Pkg; Pkg.test()'`.

## A domain and its mesh

A domain is given by its boundary, a 2π-periodic counterclockwise parametrization returning anything convertible to
`Vec2`. `discretize_curve` cuts it into panels, each a polynomial of degree ``p - 1``, bisecting until each is
resolved to `tol` and no longer than the local feature size; `build_mesh` then meshes the inside:

```julia
using OpenSurfaces

p = 10
pans = discretize_curve(θ -> Vec2(cos(θ), sin(θ)), p)       # Vector{CurvePanel}
bd, ad, info = build_mesh(pans, p)                          # BoundaryDisc, AreaDisc, details

npanels(bd), length(bd)            # 16 boundary panels, 160 nodes on Γ
npanels(ad), length(ad)            # 87 area panels, 8700 nodes (p² per panel)
```

A hole is a clockwise curve (`reverse_curve` of a counterclockwise one); several components ("drops") are a list:

```julia
outer = discretize_curve(θ -> Vec2(2cos(θ), 1.2sin(θ)), p)
hole  = reverse_curve(discretize_curve(θ -> Vec2(0.6, 0.2) + 0.5Vec2(cos(θ), sin(θ)), p))
bd, ad, _ = build_mesh(outer, p; holes = [hole])            # one drop, two curves: ncurves(bd) == 2

other = discretize_curve(θ -> Vec2(5 + cos(θ), sin(θ)), p)
bd, ad, _ = build_mesh([[outer, hole], [other]], p)         # two drops: drop_panels(ad, 2), drop_curves(bd, 1), …
```

Curves that come close to each other (a hole near the rim, two drops across a narrow gap) should be discretized
together, with `discretize_curves`, so that the panels facing the gap are short enough for it.

The area mesh has a layer of edge panels on ``\Gamma`` (one per boundary panel, its bottom edge on ``\Gamma``),
`edge_layers - 1` more layers above it graded toward ``\Gamma``, and an interior of quadrilaterals meshed by Gmsh.
`mesh_quality(ad)` reports Jacobian and aspect ratios. See [Geometry](@ref method-geometry) for the construction and its keywords.

## Nodes, weights and densities

The nodes are `ad.x` (a tensor grid of p × p Gauss points per panel), with quadrature weights `ad.w`; on ``\Gamma``,
`bd.x`, the arclength weights `bd.w`, normals `bd.n` (outward), tangents `bd.τ` and curvatures `bd.κ`.

A density on the area mesh comes in one of two forms, chosen when you build an evaluator:

- **`density = :stored`** (the default): the values `μ̃` that the mesh stores. On the panels of the first layer
  (the *weighted* panels) a density is written ``\mu = \tilde\mu\,(1 + t)^{-1/2}``, ``t = -1`` on ``\Gamma``, and
  the factor is part of the weights, so that a density with the ``d^{-1/2}`` edge singularity of a first-kind
  equation is integrated to full order: ``\int_\Omega f\mu\,dA \approx \sum_i f(x_i)\,\tilde\mu_i\,w_i``. The
  conversion is `ad.wfac`, ``\mu = \tilde\mu \cdot`` `wfac` (1 off the weighted panels). This is the form of the
  unknown of a first-kind equation.
- **`density = :smooth`**: physical values of a density smooth up to ``\Gamma``. They are integrated on an unweighted
  copy of the mesh. Integrating a smooth density with the weighted rule converges only algebraically (the area of the
  unit disc as `sum(ad.w ./ ad.wfac)` is off by 1e-4 at p = 10), so use `:smooth` for smooth data.

## An evaluator

An `Evaluator` is the potential of a kernel from sources on a mesh, at a set of targets: the plain quadrature rule
summed fast, plus sparse local corrections where the kernel is singular. Build it once; apply it to many densities.

```julia
using SpecialFunctions: ellipe
r = [hypot(x...) for x in ad.x]

E = Evaluator(LaplaceKernel3D(), ad; density = :smooth)     # 1/(4πρ), far field by FMM3D
u = E(ones(length(ad)))                                     # S[1] = E(r²)/π at the area nodes, to 1.2e-8

Eb = Evaluator(LaplaceKernel3D(), ad; density = :smooth, targets = :boundary, bd)   # at the nodes of Γ
Ep = Evaluator(LaplaceKernel3D(), ad; density = :smooth, targets = [Vec2(0, 0), Vec2(0.5, 0)])
```

`targets` is `:area` (the default), `:boundary`, a vector of points in the plane, or a tuple of these (one fast sum,
one output per set). A scalar kernel takes and returns numbers per node; a vector or tensor kernel `Vec2`s:

```julia
# the in-plane half-space Stokeslet (viscosity μ_3D below a free surface): a disc translating in its plane
F = [Vec2(1, 0) / (2π * sqrt(max(1 - sum(abs2, x), 0.0))) for x in ad.x] ./ ad.wfac   # a stored density
U = Evaluator(Stokeslet3D(), ad)(F)                        # (3/16, 0) at every node, to 4e-8
```

| Kernel | | Fast sum |
|:--|:--|:--|
| `Stokeslet3D(; μ_3D)` | in-plane block of the half-space Stokeslet, ``(I/\rho + rr^T/\rho^3)/(4\pi\mu_{3D})``; for an unbounded fluid of viscosity ``\mu_f`` pass ``\mu_{3D} = 2\mu_f`` | Ewald + NUFFT, or FMM3D |
| `LaplaceKernel3D()` | ``1/(4\pi\rho)`` | FMM3D |
| `LaplaceKernel2D()`, `LaplaceGradient2D()`, `BiharmonicGradient2D(; κ, R)` | ``\log\rho/2\pi``, its gradient, the gradient of the biharmonic kernel | FMM2D |
| `MHKernel(k)`, `MHNormalKernel(k, normals)` | ``K_0(k\rho)/2\pi`` and its normal derivative at the targets | FMM2D (Helmholtz at ``ik``) |
| `OpenSurfaces.StokesKernel(μ_2D, μ_3D)` | the 2D Stokeslet `G2` plus `G3` | direct |
| any callable `K(r)` or `K(i, r)` | your kernel | direct |
| `SymbolKernel` | a kernel from its Fourier symbol (below) | as `remainder` of its base: the base's; as the kernel itself: direct |

`fast = :direct` forces direct sums. `tol` sets the accuracy of the local corrections and `eps` that of the fast sum
(both default to `OpenSurfaces.DEFAULT_TOL` = 1e-6).

## Solving: the evaluator as an operator

With one target set, an evaluator is a linear operator on `Float64` vectors (interleaved, ``[u_{1x}, u_{1y},
u_{2x}, \dots]``, for vector and tensor kernels; `interleave`, `deinterleave`): `size(E)`, `eltype(E)`,
`mul!(y, E, x)` and `E * x`. That is what Krylov solvers take. The charged disc, the first-kind equation
``\int_\Omega \sigma(y)/(4\pi|x - y|)\,dA_y = 1`` on the unit disc, with GMRES from
[Krylov.jl](https://github.com/JuliaSmoothOptimizers/Krylov.jl):

```julia
using Krylov, LinearAlgebra
p = 8
bd, ad, _ = build_mesh(discretize_curve(θ -> Vec2(cos(θ), sin(θ)), p), p; edge_layers = 1)   # one layer: see below
E = Evaluator(LaplaceKernel3D(), ad)               # stored density: the unknown has the edge singularity
σ̃, stats = gmres(E, ones(length(ad)); rtol = 1e-8)
sum(σ̃ .* ad.w)                                     # the total charge, 8 (the capacitance of the disc)
σ = σ̃ .* ad.wfac                                   # physical values, 4/(π√(1 − r²))
```

That takes 48 iterations. The mesh has one edge layer because the unknown has the ``d^{-1/2}`` singularity, which
the graded layers do not represent ([Limits](@ref "Graded layers and edge-singular densities")); with the default
three, GMRES needs 114 iterations, and a dense solve at p = 10 is within 2.8e-5 near the edge instead of 1.9e-10.
`local_blocks(E, panel_groups(ad))` gives the dense diagonal blocks of the operator on groups of panels (each edge
column with its graded layers, each interior panel alone); a block-Jacobi preconditioner built from them brings it
to 31:

```julia
idx, blocks = local_blocks(E, panel_groups(ad))
F = lu.(blocks)
struct BlockJacobi end
Base.size(::BlockJacobi) = size(E); Base.eltype(::BlockJacobi) = Float64
LinearAlgebra.mul!(y, ::BlockJacobi, x) = (for (i, f) in zip(idx, F); y[i] = f \ x[i]; end; y)
σ̃, stats = gmres(E, ones(length(ad)); M = BlockJacobi(), ldiv = false, rtol = 1e-8)
```

`evaluator_matrix(E)` is the dense matrix (for small problems, and for checks).

## Sources on the boundary

The same evaluator takes sources on ``\Gamma``: `Evaluator(K, bd; targets)`, with targets off ``\Gamma`` or on it
(on it: kernels bounded at ``r = 0``, and those whose logarithmic split ``A\log\rho + S`` is known, `LaplaceKernel2D`,
`MHKernel` and the 2D Stokeslet). Its sum is direct, but for `MHKernel` (FMM2D). For the 2D Stokes single layer on ``\Gamma`` there are also the dense operators of a boundary
integral equation:

```julia
lq = LineQuad(bd)                       # near lists and rules for Γ
S = slp2d_velocity_matrix(lq)           # 2N_Γ × 2N_Γ, the single layer restricted to Γ
T = traction_self_matrix(bd)            # its interior traction on Γ (jump ½)
u = slp2d_eval(lq, [Vec2(0.3, 0.1)], ν) # at points off Γ, ν::Vector{Vec2} at the nodes of Γ
```

## Kernels of your own

A kernel is any callable `K(r)` of `r = x - y` (target minus source), or `K(i, r)` when it depends on the target
`i` (a traction with the normal at each target). It returns a number (a scalar kernel) or a value convertible to
`Mat2`. A kernel that is not scalar declares its shape, so the evaluator knows which slots of the 2×2 form the
potential uses:

```julia
struct MyTensor end
(::MyTensor)(r) = (ρ = hypot(r...); Mat2(1/ρ, 0, 0, 1/ρ) / 4π)
OpenSurfaces.kernel_shape(::MyTensor) = (2, 2)       # (1, 1) scalar (the default), (2, 1) vector, (2, 2) tensor
E = Evaluator(MyTensor(), ad)                        # direct sums, local corrections from the quadrature
```

A plain closure works for a scalar kernel; for any other shape wrap it in a type as above (the evaluator probes the
kernel and refuses a 2×2 form it would truncate). The quadrature handles any integrable singularity at ``r = 0``
(``\log\rho``, ``1/\rho``, and their products with smooth functions).

## Kernels from Fourier symbols

A radial kernel known by its Fourier symbol ``\hat s(k)`` is a standard kernel with the same singularity plus a smooth
remainder. `SymbolKernel` tabulates the remainder by Hankel transforms; an evaluator then keeps the standard kernel's
fast sum and corrections and adds the remainder. The 3D Laplace kernel over a layer of depth ``H`` with the value
fixed on its floor has symbol ``\tanh(kH)/(2k)``:

```julia
H = 0.3
K = SymbolKernel(k -> tanh(k * H) / (2k); base = LaplaceKernel3D(), scale = H, outputs = (:value,),
                 remainder = k -> -1 / (k * (exp(2k * H) + 1)))      # the same difference, written stably
S = SingleLayer3D(ad; remainder = K)                                 # the operator of 1/(4πρ) + remainder
```

See [Kernels from Fourier symbols](@ref method-symbols) for the conditions on the base and the remainder.
