# [Quadrature](@id method-quadrature)

The local corrections replace the plain rule on the panels near a target by weights that integrate the singular
kernel: for each near (target, panel) pair, the accurate weights minus the plain ones. They are assembled once into a
sparse matrix (`area_correction`, `line_correction`) in an interleaved 2×2 layout. The rules are built in the
panel's parameter space and treat the kernel as a black box ``K(r)`` or ``K(i, r)`` returning a 2×2 matrix (a
scalar kernel is carried in one slot), so they serve every kernel, including yours.

The rules depend on the geometry only (the target and the panel), not on the kernel, and most of a pass is spent
building them and contracting with the Lagrange basis. Several scalar kernels on the same targets therefore share one
pass: `area_correction(ad, K; ncomp = M)` takes a kernel returning ``M`` components (an `SVector{M}`, a tuple, or a
number for ``M = 1``) and returns ``M`` scalar corrections (``N_t \times N`` each, one sparsity pattern), each the
same as that of its component alone. `Evaluator((K1, K2, …), ad; …)` builds the evaluators of several kernels on the
same mesh, targets and density from one such pass (identical to building them one at a time).

## Area sources

For a target ``x`` and an area panel, the weights ``W_m`` (one 2×2 block per node of the panel) satisfy

```math
\int_{\text{panel}} K(x - y)\,\mu(y)\,dA_y \approx \sum_m W_m\,\tilde\mu_m,
\qquad W_m = \sum_q K(x - y_q)\,\ell_m(s_q, t_q)\,\omega_q\,J(s_q, t_q),
```

``\ell_m`` the tensor Lagrange basis of the panel's nodes and ``(s_q, t_q, \omega_q)`` a rule adapted to the
target.

**Near panels.** A (target, panel) pair is corrected when ``|x - c_k| <`` `near_factor` ``\cdot R_k`` (``c_k`` the
panel's center, ``R_k`` its circumradius). By default `near_factor` follows from ``p`` and `tol`, so that the
``p \times p`` rule is accurate to about `tol` outside, and is at least 1.2 (targets next to a neighboring panel's
corner).

**Targets on the panel** (at a node, on a side, at a corner, on ``\Gamma``): a polar rule with its apex at the
target. The square is cut into up to four triangles from the apex to its sides, each mapped to ``(u, v) \in [0,1]^2``
with Jacobian ``u|\det|``, which cancels ``1/\rho``. In ``u`` a composite rule clusters points at the apex
(``u = u_1 w^4`` on ``[0, u_1]``, which also integrates ``u\log u``); in ``v`` a composite Gauss–Legendre rule is
refined toward the complex singularities of the kernel along the base, found by linearizing at the apex and by a
complex Newton solve on the far side, so apexes arbitrarily close to a side are handled. On a weighted panel the
coordinates remove the ``(1 + t)^{-1/2}`` weight exactly: ``t = -1 + 2\tau^2`` above the edge, and for a target on
``\Gamma`` itself the weight factors as ``(2uv)^{-1/2}`` or ``(2u)^{-1/2}``, absorbed by the radial rule and by
``v = z^2``. The rule sizes have floors set by the degree of the Lagrange basis along each line (``p - 1`` in ``s``
and ``t``, up to ``3(p - 1)`` along a radial line on a weighted panel), so the basis is integrated to about
`tol`/10 as well as the kernel.

**Targets off the panel**: an adaptive quadtree in ``(s, t)``. A box is accepted with a ``q \times q`` tensor rule,
``q`` from a Bernstein-ellipse estimate of the target's distance; boxes touching ``t = -1`` on a weighted panel use
Gauss–Jacobi in ``t``.

Targets given as points are located on the panels near them by a Newton solve of ``x(s, t) = x_0``
(`panel_preimage`); targets on ``\Gamma`` can be declared (`boundary_target_info`). With an `AreaUpsampling` the far
field uses a finer ``q \times q`` rule on every panel, which shrinks the near zone.

## Line sources

For a density ``\nu`` on ``\Gamma`` and a kernel ``K`` (the 2D Stokeslet ``G_2``, for instance),

```math
v(x) = \oint_\Gamma K(x - y)\,\nu(y)\,ds_y.
```

**Near panels.** Panel ``k`` is near ``x`` when ``|x - c_k| < \beta L_k`` (``c_k = \gamma_k(0)``, ``L_k`` its
length), ``\beta`` chosen from the convergence rate of the ``p``-point Gauss–Legendre rule and `tol`.

**Targets off ``\Gamma``**: adaptive bisection in the panel parameter with ``q``-point Gauss–Legendre leaves. A leaf
``[a, b]`` is accepted when ``|x - \gamma(m)| \ge \alpha\,|\gamma'(m)|\,(b - a)``, ``m`` its midpoint, which keeps
the nearest complex singularity of the integrand outside a Bernstein ellipse of parameter about 3.7.

**Targets on ``\Gamma``** (at a node or between nodes): the logarithm is split off,
``-\log|r(s)| = -\log|s_i - s| - \log(|r(s)|/|s_i - s|)``. The first part is integrated exactly against Legendre
polynomials on an upsampled Gauss–Legendre grid (log moments in closed form, `log_product_weights`); the second and
the smooth parts of the kernel use the plain upsampled rule. This needs the kernel's split ``A\log\rho + S``, known
for `LaplaceKernel2D`, `MHKernel` and the 2D Stokeslet. A kernel bounded at ``r = 0`` (``\rho\log\rho``) keeps the
adaptive weights; an unbounded kernel without a split is refused.

**The traction.** The interior traction of the 2D Stokes single layer on ``\Gamma`` is

```math
T_{\Gamma\Gamma}\nu(x) = \tfrac12\nu(x) + \mathrm{PV}\!\oint_\Gamma T_2(x - y, n(x))\,\nu(y)\,ds_y,
\qquad T_2(r, n)f = -\frac{1}{\pi}\frac{r\,(r\cdot n)(r\cdot f)}{\rho^4},
```

whose kernel is smooth along ``\Gamma`` with the diagonal limit ``-(\kappa/2\pi)\,\tau\tau^T``, so the plain rule
with that diagonal value is spectrally accurate (`traction_self_matrix`; `jump = -0.5` for the exterior limit).
