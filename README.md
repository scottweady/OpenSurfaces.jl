# OpenSurfaces.jl

Singular integrals on flat open surfaces Ω ⊂ R² (the plane z = 0) and on their edges Γ = ∂Ω: boundary panels and
area meshes with graded edge layers, the standard kernels and kernels built from Fourier symbols, local quadrature
corrections for singular and near-singular targets, fast sums (FMM, Ewald with NUFFT, smooth convolution), and the
evaluators built from them. Nothing here knows a physical model: a model's kernels (a finite-depth subphase, an
embedded membrane) are passed in as callables or as `SymbolKernel`s.

The kernel contract (what a callable kernel takes and returns, `r = x − y`, coincident points) is in the module
docstring, `?OpenSurfaces`. Every solve, fast sum and quadrature takes `tol`, defaulting to
`OpenSurfaces.DEFAULT_TOL` = 1e-6.

## Documentation

https://scottweady.github.io/OpenSurfaces.jl/ (getting started, the method, examples, the API reference and a code
map), built from `docs/` and published to the gh-pages branch on every push to main. To build it locally:

```bash
julia --project=docs -e 'using Pkg; Pkg.instantiate()'    # once
julia --project=docs docs/make.jl                          # then open docs/build/index.html
```

## Use

A domain from a 2π-periodic counterclockwise parametrization, its mesh, and a potential on it (from this directory, after
`julia --project -e 'using Pkg; Pkg.instantiate()'` once, `julia --project`):

```julia
using OpenSurfaces

# the unit disc at p = 10: boundary panels, then the boundary nodes and the area mesh
pans = discretize_curve(θ -> Vec2(cos(θ), sin(θ)), 10)
bd, ad, _ = build_mesh(pans, 10)

# the 3D Laplace single layer of a unit density on the disc: S[1] = E(r²)/π (complete elliptic integral)
S = SingleLayer3D(ad)
u = single_layer(S, ones(length(ad)))       # within 2e-8 of E(r²)/π at the default tol
```

| | |
|:--|:--|
| Geometry | `Vec2`, `Mat2`, panel rules (`gl_rule`, `gj_rule`), `CurvePanel`, `discretize_curve`, `discretize_curves`, `reverse_curve`, `BoundaryDisc`, `build_mesh`, `film_drops`, `AreaDisc`, `mesh_quality`, `local_feature_size`, `interpolate_nodal` |
| Kernels | `G2`, `G3`, `P2`, `T2`, `stress2`, `LaplaceKernel2D`, `LaplaceGradient2D`, `BiharmonicGradient2D`, `MHKernel`, `MHNormalKernel`; from Fourier symbols: `SymbolKernel`, `Stokeslet3D`, `LaplaceKernel3D`, `kernel_symbol`, `kernel_remainder`, `cached_kernel`. `G3` and `Stokeslet3D(; μ_3D)` are the in-plane block of the **half-space** Stokeslet (viscosity μ_3D below a free surface), (1/(4πμ_3D))(I/ρ + r rᵀ/ρ³): for an unbounded fluid of viscosity μ_f, pass μ_3D = 2μ_f |
| Fast sums | `stokes2d_apply`, `stokes3d_plane_apply` (FMM), `EwaldStokes3D`, `EwaldLaplace3D`, `RadialEwald2D`, `FilmEwald` (2D + 3D on one grid), `SmoothConvolution` |
| Quadrature | `LineQuad` and the boundary single layer and traction (`slp2d_*`, `traction_self_matrix`), `area_correction`, `area_local_weights` |
| Evaluators | `Evaluator` (any kernel, area or line sources, any targets), `VolumePotentials`, `SingleLayer3D`; as operators: `size`, `mul!` and `*` on interleaved `Float64` vectors (for Krylov solvers), `evaluator_matrix` (dense, every kernel shape), `panel_groups` and `local_blocks` (diagonal blocks, for a block-Jacobi preconditioner) |

The standard kernel types the quadrature dispatches on, `StokesKernel` and `TractionKernel`, are not exported
(`OpenSurfaces.StokesKernel`). Names starting with `_` are internal and may change.

## Layout

| Path | |
|:--|:--|
| `src/OpenSurfaces.jl` | module docstring (kernel contract), `DEFAULT_TOL`, exports, includes, the FFTW threads fix in `__init__` |
| `src/geometry/` | `panels.jl` (core types, panel rules), `boundary.jl` (`discretize_curve`), `mesh.jl` (`build_mesh`, by Gmsh), `interpolate.jl` |
| `src/kernels/` | closed forms, Laplace and modified Helmholtz kernels, FMM wrappers, `symbol.jl` (`SymbolKernel`), Ewald sums (`ewald.jl`, `ewald2d.jl`) |
| `src/quadrature/` | `line_quad.jl` (boundary), `area_quad.jl` (area, graded edge layers) |
| `src/evaluators/` | `evaluator.jl` (`Evaluator`), `volume.jl`, `single_layer.jl` |
| `test/` | regression tests at the production settings against exact or direct references; `common/` shared references; `validation/` the convergence sweeps and timings |
| `docs/` | the documentation: `make.jl`, `src/` (the pages), `figures/` (data and plots), `codemap.jl` |
| `examples/` | scripts that evaluate or solve against exact solutions (`julia -t 8 --project examples/<name>.jl`) |

## Tests

From this directory:

```bash
julia -t 8 --project -e 'using Pkg; Pkg.test()'
```

`Pkg.test(test_args=["kernels", "symbol"])` runs only test_kernels.jl and test_symbol.jl. The validation tier is opt-in
and needs QuadGK besides OpenSurfaces (head of test/validation/run_all.jl):

```bash
julia -t 8 -e 'using Pkg; Pkg.activate(temp=true); Pkg.develop(path="."); Pkg.add("QuadGK"); include("test/validation/run_all.jl")'
```

Dependencies: FFTW, FINUFFT, FMM2D, FMM3D, Gmsh, SpecialFunctions, StaticArrays, LinearAlgebra, SparseArrays.
