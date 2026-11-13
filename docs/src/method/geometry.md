# [Geometry](@id method-geometry)

## Conventions

``\Omega`` lies in the plane ``z = 0``; points are `Vec2`. Its boundary ``\Gamma`` is oriented with ``\Omega`` on
the left: outer curves counterclockwise, holes clockwise. ``n`` is the outward unit normal (out of ``\Omega``, so
into a hole), ``\tau`` the unit tangent, ``n = (\tau_2, -\tau_1)``, and the curvature ``\kappa > 0`` where
``\Omega`` is locally convex.

## Boundary panels

`discretize_curve(γ, p)` cuts a 2π-periodic parametrization ``\gamma(\theta)`` into panels. Each panel is the
polynomial of degree ``p - 1`` interpolating ``\gamma`` at ``p`` Gauss–Legendre nodes of its interval in
``\theta``, stored as Legendre coefficients (`CurvePanel`); only ``\gamma`` itself is needed, not its derivatives. A
panel is bisected in ``\theta`` until

- it is resolved: the norm of its last two Legendre coefficients is below `tol` × its length (with a floor of
  1000 ε × the curve's size, for rounding);
- it is no longer than `maxlen`, than `lencap(x)` at its midpoint if that function is given, and than `lfs_factor`
  × the local feature size;
- neighboring panels differ in length by at most a factor 2 (`balance`).

The local feature size (`local_feature_size`) combines curvature and thickness,

```math
\mathrm{lfs}(x) = \min\Big(\frac{1}{|\kappa(x)|},\ \tfrac12 \min\{|x - y| : y \in \Gamma,\ n(x)\cdot n(y) < 0\}\Big),
```

the second term half the distance to the nearest part of ``\Gamma`` facing away from ``x``: across a neck of
half-width ``\delta`` it is about ``\delta``, and it also sees two parts of ``\Gamma`` approaching from outside.
`discretize_curve` sees one curve at a time; `discretize_curves` discretizes the curves of a domain together and
also limits each panel to `gap_factor` × its distance to the other curves (a hole near the rim, a narrow ring).

The boundary discretization (`BoundaryDisc`) holds the Gauss–Legendre nodes of every panel, their arclength
weights, tangents, normals and curvatures.

## The area mesh

Every area panel is a ruled quadrilateral over ``(s, t) \in [-1, 1]^2``,

```math
x(s, t) = \tfrac12(1 - t)\,B(s) + \tfrac12(1 + t)\,T(s),
```

with bottom and top curves ``B``, ``T`` stored as Legendre coefficients, oriented so that
``J = \det[\partial_s x\ \partial_t x] > 0``, and a ``p \times p`` tensor grid of nodes (``s`` fastest). The
quadrature weight of a node is ``w = w_s w_t J``. `build_mesh(panels, p)` builds the mesh in three parts.

**The edge layer.** One edge panel sits on each boundary panel: its bottom ``B`` is the boundary panel's curve, so
``t = -1`` lies exactly on ``\Gamma``, and its top is the straight segment between inner vertices
``w_k = v_k - h_k n_k`` (``v_k`` the start of boundary panel ``k``). The strip width is

```math
h_k = \min\big(\text{strip\_factor} \cdot \min(L_{k-1}, L_k),\ \text{thick\_factor} \cdot \mathrm{lfs}(v_k)\big),
```

at least 1.5 times the inward bulge of the adjacent panels beyond their chords, and is then shrunk locally until
every edge panel has ``\min J > \text{jtol} \cdot \max J``, no top segment collapses, and the inner polygon
``W = (w_1, \dots, w_N)`` is simple, does not cross ``\Gamma`` and keeps a clearance from its other parts. These
conditions guarantee that the edge panels tile the strip between ``\Gamma`` and ``W``; when the repair fails, the
error asks for a finer boundary.

**The weight at the edge.** The solution of a first-kind integral equation on an open surface (the charge on a
conducting disc, the traction on a rigid plate) behaves like ``d^{-1/2}`` at the edge, ``d`` the distance to
``\Gamma``. On the edge panels a density is written

```math
\mu = \tilde\mu\,(1 + t)^{-1/2},
```

with ``\tilde\mu`` smooth: the nodes store ``\tilde\mu``, the ``t``-rule is the Gauss–Jacobi rule of weight
``(1 + t)^{-1/2}`` (from the ``2p``-point Gauss–Legendre rule by ``t = 2\sigma^2 - 1``), and its weight is part of
``w``. These panels are *weighted*; `ad.wfac` is ``(1 + t)^{-1/2}`` on them and 1 elsewhere.

**Graded layers.** With `edge_layers = L > 1` (default 3) each edge panel is split in ``t`` into ``L`` layers, with
interfaces at ``(1 + t)/2 = \sigma^{L-1}, \dots, \sigma`` (``\sigma`` = `edge_grading`, default 0.2), geometrically
graded toward ``\Gamma``. Only the layer on ``\Gamma`` is weighted. The layers resolve the ``d^{1/2}\log d`` terms
of a density that is bounded at the edge; a ``d^{-1/2}`` density is better served by one layer
([Limits](@ref "Graded layers and edge-singular densities")). All the layers over boundary panel ``k`` form
its *edge column* (`bpanel = k`).

**The interior.** The polygon ``W`` is meshed by Gmsh (Delaunay, with the target edge length
``f(x) = \min(\text{interior\_size}, \min_k(\text{size\_factor}\,\ell_k + \text{grading}\,|x - w_k|))``,
``\ell_k`` the length of the edges of ``W`` at ``w_k``) and recombined into quadrilaterals (Blossom matching);
leftover triangles and non-convex quadrilaterals are split into three quadrilaterals each. With
`coarse_interior = true` every other vertex of ``W`` is moved onto the segment of its neighbors, so the interior
has half as many boundary edges and coarser panels next to the strip (non-conforming there, still an exact tiling).

A domain with holes is one *drop* with several curves; several drops are meshed one by one and concatenated (drop
``i`` owns `drop_panels(ad, i)`; its curves are `drop_curves(bd, i)`, the outer first). `mesh_quality` reports the
Jacobian ratio and aspect ratio of every panel.

## Smooth densities

A density that is smooth up to ``\Gamma`` (a source term, a concentration) does not have the ``d^{-1/2}`` factor,
and integrating it as ``\mu\,(1 + t)^{1/2}`` with the weighted rule converges only algebraically. With
`density = :smooth` an evaluator integrates it on an unweighted copy of the mesh instead (Gauss–Legendre in ``t`` on
every panel; on the weighted panels the density is interpolated from their Gauss–Jacobi nodes). `interpolate_nodal`
evaluates the panel interpolant of nodal values at any points.
