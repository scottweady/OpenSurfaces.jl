# API reference

```@docs
OpenSurfaces
```

## Geometry

### Points, rules and Legendre series

```@docs
Vec2
Mat2
Rule1D
gl_rule
gj_rule
lagrange_row
lagrange_row!
interp_matrix
diff_matrix
legendre_P
legendre_eval
legendre_coefs
legendre_deriv_coefs
segment_coefs
```

### Boundary curves

```@docs
CurvePanel
curve_eval
discretize_curve
discretize_curves
reverse_curve
local_feature_size
curve_area
curve_length
panel_length
```

### Area meshes

```@docs
build_mesh
film_drops
mesh_quality
Panel
panel_map
panel_diameter
panel_center
panel_interp_matrix
interpolate_nodal
```

### Discretizations, drops and curves

```@docs
BoundaryDisc
AreaDisc
npanels
panel_nodes
trule
ndrops
drop_panels
drop_nodes
panel_drop
node_drop
ncurves
curve_panels
curve_nodes
drop_curves
cyclic_maps
```

## Kernels

### Closed forms

```@docs
G2
G3
P2
T2
stress2
LaplaceKernel2D
LaplaceGradient2D
BiharmonicGradient2D
MHKernel
MHNormalKernel
```

The standard kernel types the quadrature dispatches on (not exported):

```@docs
OpenSurfaces.StokesKernel
OpenSurfaces.TractionKernel
```

### Kernels from Fourier symbols

```@docs
SymbolKernel
Stokeslet3D
LaplaceKernel3D
kernel_symbol
kernel_remainder
remainder_tensor
kernel_integral
table_range
RadialTable
cached_kernel
```

## Fast sums

```@docs
stokes2d_apply
stokes2d_direct
stokes3d_plane_apply
stokes3d_plane_direct
interleave
deinterleave
EwaldStokes3D
ewald_apply
FilmEwald
film_ewald_apply
ewald_spread
ewald_velocity
ewald_traction
ewald_velocity3d_boundary
SmoothConvolution
smooth_apply
```

## Quadrature

### On the boundary

```@docs
LineQuad
line_near_lists
line_panel_weights!
line_correction
line_potential_eval
line_smooth_matrix
log_product_weights
slp2d_correction
slp2d_self_correction
slp2d_velocity_matrix
slp2d_eval
slp2d_traction_eval
traction_self_matrix
```

### On the area mesh

```@docs
area_correction
area_local_weights
AreaQuadOpts
AreaUpsampling
upsample
boundary_target_info
panel_preimage
```

## Evaluators

```@docs
Evaluator
kernel_shape
evaluator_matrix
local_blocks
panel_groups
VolumePotentials
volume_grad
volume_bgrad
volume_potential
SingleLayer3D
single_layer
```
