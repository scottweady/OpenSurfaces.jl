# Limits

## Flat surfaces, targets in the plane

``\Omega`` lies in the plane ``z = 0`` and every target is in that plane: on the mesh, on ``\Gamma``, or at points
``(x, y, 0)``. There is no rule for targets off the plane (a field above the surface), and no curved surfaces.

## Smooth edges

The weight ``(1 + t)^{-1/2}`` of the edge layer is the singularity of a first-kind density at a smooth edge.
``\Gamma`` should be smooth: at a corner the density's singularity has another exponent, which the weight does not
represent, and `discretize_curve` refines toward the corner (up to `maxpanels`) without resolving it.

## Graded layers and edge-singular densities

The graded layers (`edge_layers > 1`) resolve the ``d^{1/2}\log d`` terms of a density that is bounded at the edge.
A density with the ``d^{-1/2}`` singularity of a first-kind equation is weighted only on the first, thin layer, and
continues into the unweighted layers above it, where the plain rule does not represent it. For such densities use
one layer: the charged disc at p = 10 is within 1.9e-10 near the edge (``d < 0.05``) with `edge_layers = 1`, and
2.8e-5 with the default 3 (its total charge within 4.2e-11 and 3.4e-9).

## Densities: stored or smooth

A density with the ``d^{-1/2}`` edge singularity goes in the stored form (`density = :stored`), a density smooth up to
``\Gamma`` in the smooth form (`density = :smooth`); either one in the other form converges only algebraically
([Getting started](@ref "Nodes, weights and densities")).

## Close approach

Where two parts of ``\Gamma`` come within a gap ``g`` (a thin neck, a hole near the rim, two drops), the panels facing
the gap must be shorter than about ``g``, and the strip of edge panels must fit in it. `discretize_curve` limits the
panels by the local feature size of one curve; between curves, use `discretize_curves`, since `build_mesh` meshes
several drops independently. When the strip cannot be repaired, `build_mesh` stops with an error that asks for a finer
boundary.

## Kernels

- A kernel that is not scalar declares its shape (`kernel_shape`); a closure can only be scalar. The evaluator probes
  the kernel and refuses a 2×2 value it would truncate.
- Line sources with targets on ``\Gamma`` need a kernel bounded at ``r = 0`` or one whose split ``A\log\rho + S``
  is known (`LaplaceKernel2D`, `MHKernel`, the 2D Stokeslet); any other is refused.
- `Stokeslet3D` and `G3` are the **half-space** Stokeslet (a fluid of viscosity ``\mu_{3D}`` below a free surface):
  for an unbounded fluid of viscosity ``\mu_f``, pass ``\mu_{3D} = 2\mu_f``.
- Kernels other than those of the [Fast sums](@ref method-fast-sums) table are summed directly, ``O(NM)``.

## Kernels from symbols

- The base is given, not extracted from the symbol, and must match it at both ends of ``k``.
- A remainder that decays algebraically needs a large ``k_{\max}``: a screened kernel ``e^{-\kappa\rho}/4\pi\rho``
  needs ``k_{\max} \sim 1/(\text{tol}\,\kappa)``, so it warns at the default tolerance. The Ewald sums refuse such a
  remainder (their long-range multiplier would sample its cone at ``\rho = 0``).
- `regularize` (a remainder not integrable at ``k = 0``) applies to the scalar `:gradient` output only.

## Fast sums

- `FilmEwald` skips only the area self pairs: its area nodes must be distinct and no boundary node may sit on an area
  node (true of `build_mesh`, whose area nodes are interior Gauss points); it throws otherwise.
- An `Evaluator` of `Stokeslet3D` takes one target set, and a remainder with Ewald (the default) or direct sums, not
  with FMM3D.
- The accuracy of an Ewald plan or of a remainder's local sum is fixed when it is built; only the FMMs take `eps` per
  call.
- `evaluator_matrix` is dense, ``O(N^2)`` memory: at p = 10 the unit disc has 8700 nodes, a 600 MB matrix for a
  scalar kernel. Use it for small problems and checks; solve with the operator.
