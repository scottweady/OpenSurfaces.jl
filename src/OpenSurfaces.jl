"""
    OpenSurfaces

Singular integrals on flat open surfaces Ω ⊂ R² (the plane z = 0) and on their edges Γ = ∂Ω: the
geometry and meshes, the standard kernels and kernels built from Fourier symbols (`SymbolKernel`), the local
quadrature corrections, the fast sums (FMM, Ewald + NUFFT, smooth convolution) and the evaluators built from them
(`Evaluator`, `VolumePotentials`, `SingleLayer3D`). Nothing here knows a physical model: a model passes its own
kernels in, as callables or as `SymbolKernel`s (a finite depth, an embedded membrane).

Kernel contract:
- A point kernel is a callable `K(r)`, or `K(i, r)` when it depends on the target `i` (as `TractionKernel`
  does through its normals). The quadrature accepts any callable (`_wrap_kernel` checks for `K(i, r)`).
- It returns a value convertible to `Mat2`. Several scalar kernels may share the slots if documented (the
  volume potentials use row 1).
- `r = x − y` is target minus source. Callers exclude coincident points, unless a kernel is documented as
  regular there.
- Closed forms with parameters (`G2(r; μ_2D)`, `G3(r; μ_3D)`) and two-argument forms (`T2(r, n)`,
  `stress2(r, f)`) are functions; the standard kernels the quadrature and the evaluators dispatch on are types
  (`StokesKernel`, `TractionKernel`).
- Model kernels live with their formulations (in the code that uses this package) and are passed in.
"""
module OpenSurfaces

using LinearAlgebra, SparseArrays, StaticArrays
import FMM2D, FMM3D, FINUFFT, FFTW
using SpecialFunctions: erf, erfc, erfcinv, expint
import SpecialFunctions

# Default accuracy of every solve, fast sum, quadrature and curve fit: the building blocks take `tol` (and
# fast-sum `eps`) defaulting to it.
const DEFAULT_TOL = 1e-6

# geometry: core types, boundary curves, area mesh
include("geometry/panels.jl")
include("geometry/boundary.jl")
include("geometry/mesh.jl")
include("geometry/interpolate.jl")
# point kernels and fast sums
include("kernels/kernels.jl")
include("kernels/laplace.jl")
include("kernels/helmholtz.jl")
include("kernels/fmm.jl")
include("kernels/symbol.jl")
include("kernels/ewald2d.jl")
include("kernels/ewald.jl")
include("kernels/ewald_radial2d.jl")
# local (singular / near-singular) quadrature
include("quadrature/line_quad.jl")
include("quadrature/area_quad.jl")
# evaluators: the common evaluator, and potentials of bounded densities on the film
include("evaluators/evaluator.jl")
include("evaluators/volume.jl")
include("evaluators/single_layer.jl")

# FINUFFT links the same libfftw3 as FFTW.jl, which (with Julia threads > 1) installs a callback that runs FFTW's
# parallel loops as Julia tasks. FINUFFT's threaded FFTs then segfault inside those tasks once the grids are large
# enough for FFTW to thread them (a dumbbell-shaped film at tol 1e-10: 4 of 4 runs).
# Removing the callback lets FFTW use its own threads: no crash, same time at tol 1e-6. FFTW.jl's own plans here
# are single-threaded (`FFTW.get_num_threads() == 1`). The callback is process-wide, so this applies to the session.
# FFTW.jl opens libfftw3 and libfftw3f lazily, and the first use of either installs the callback for both, so both
# are opened before the callback is removed.
function __init__()
    if FFTW.fftw_provider == "fftw" && Threads.nthreads() > 1
        FFTW.dlopen(FFTW.libfftw3); FFTW.dlopen(FFTW.libfftw3f)
        ccall((:fftw_threads_set_callback, FFTW.libfftw3), Cvoid, (Ptr{Cvoid}, Ptr{Cvoid}), C_NULL, C_NULL)
        ccall((:fftwf_threads_set_callback, FFTW.libfftw3f), Cvoid, (Ptr{Cvoid}, Ptr{Cvoid}), C_NULL, C_NULL)
    end
end

# The public names; everything else is private (`OpenSurfaces._x`).
# geometry
export Vec2, Mat2, Rule1D, gl_rule, gj_rule, lagrange_row, lagrange_row!, interp_matrix, diff_matrix, legendre_eval,
       legendre_coefs, legendre_deriv_coefs, legendre_P, segment_coefs, CurvePanel, curve_eval, Panel, panel_map,
       panel_diameter, panel_center, drop_panels, drop_nodes, panel_drop, node_drop, ncurves, curve_panels,
       curve_nodes, drop_curves, cyclic_maps, BoundaryDisc, AreaDisc, npanels, panel_nodes, trule,
       panel_interp_matrix, discretize_curve, discretize_curves, reverse_curve, build_mesh,
       film_drops, mesh_quality, local_feature_size, curve_area, curve_length, panel_length, ndrops,
       interpolate_nodal
# kernels and fast sums
export G2, G3, P2, T2, stress2, LaplaceKernel2D, LaplaceGradient2D, BiharmonicGradient2D, MHKernel, MHNormalKernel,
       stokes2d_apply, stokes2d_direct, stokes3d_plane_apply, stokes3d_plane_direct,
       interleave, deinterleave, SmoothConvolution, smooth_apply, EwaldStokes3D, EwaldLaplace3D, RadialEwald2D, ewald_apply, FilmEwald,
       film_ewald_apply, ewald_spread, ewald_velocity, ewald_traction, ewald_velocity3d_boundary
# kernels from Fourier symbols
export SymbolKernel, RadialTable, Stokeslet3D, LaplaceKernel3D, kernel_symbol, kernel_remainder, remainder_tensor,
       table_range, kernel_integral, cached_kernel
# quadrature
export LineQuad, line_near_lists, line_panel_weights!, line_correction, slp2d_correction, slp2d_self_correction,
       slp2d_velocity_matrix, traction_self_matrix, line_smooth_matrix, line_potential_eval, slp2d_eval,
       slp2d_traction_eval, log_product_weights, AreaUpsampling, upsample, area_correction, area_local_weights,
       AreaQuadOpts, boundary_target_info, panel_preimage
# evaluators
export Evaluator, evaluator_matrix, local_blocks, panel_groups, kernel_shape, VolumePotentials, volume_grad, volume_bgrad, volume_potential,
       SingleLayer3D, single_layer

end
