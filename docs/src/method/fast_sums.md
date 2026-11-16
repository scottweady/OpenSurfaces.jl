# [Fast sums](@id method-fast-sums)

The plain rule is a sum over all (target, source) pairs, ``u_i = \sum_j K(x_i - y_j)\,f_j`` with the weights in
``f``. Every fast sum here skips coincident pairs (a target exactly on a source), as the local corrections expect.
An `Evaluator` picks the sum by the kernel's type:

| Kernel | Sum |
|:--|:--|
| `Stokeslet3D` | Ewald + NUFFT (`EwaldStokes3D`; default), or FMM3D (`fast = :fmm`) |
| `LaplaceKernel3D` | Ewald + NUFFT (`EwaldLaplace3D`; default), or FMM3D (`fast = :fmm`) |
| `LaplaceKernel2D`, `LaplaceGradient2D`, `BiharmonicGradient2D` | FMM2D (Laplace; default), or Ewald + NUFFT (`RadialEwald2D`, `fast = :ewald`) |
| `MHKernel`, `MHNormalKernel` | FMM2D (Helmholtz at the imaginary wavenumber ``ik``; default), or Ewald + NUFFT (`RadialEwald2D`, `fast = :ewald`) |
| `StokesKernel(μ_2D, μ_3D)` (G2 + G3) | Ewald + NUFFT (`EwaldStokes3D` with `μ_2D`; default) |
| `SymbolKernel` of base `Stokeslet3D` or `LaplaceKernel3D` | Ewald + NUFFT of the whole kernel (default) |
| `BiharmonicGradient2D` + a `remainder` with a far field | Ewald + NUFFT of the whole kernel (`RadialEwald2D`, `out = :bgrad`; default) |
| `LaplaceKernel3D` or `BiharmonicGradient2D` + a `remainder` without one | local sum over the pairs closer than ``\rho_c`` |
| any other | direct |

With few targets the default is the direct sum instead, for every kernel above summed by Ewald by default: an Ewald
sum costs about the same whatever the number of targets (its grid grows with the sources), while the direct sum
grows with sources × targets, so below a number of targets the direct sum wins. That number grows with the
accuracy; `fast = :auto` takes the direct sum below

```math
M^*(\text{eps}) = 2000\,\bigl(\log_{10}(1/\text{eps})/6\bigr)^{3.5}
```

targets in all (about 2000 at eps 1e-6, 5500 at 1e-8, 12000 at 1e-10), the crossovers measured on 13.9k sources and
16 threads (1000–2000 targets at 1e-6, 8000–20000 at 1e-10 across these kernels). A boundary's nodes as targets are
then summed directly. `fast = :ewald` (or a `plan`) keeps Ewald.

The defaults follow timings on 16 threads (studies/laplace_ewald). For the 3D Laplace kernel an Ewald apply is 4–15
times faster than FMM3D from 8.8k to 410k nodes (0.058 s against 0.6 s at 110k nodes, eps 1e-6), and its setup costs
about one FMM3D apply. For the 2D kernels FMM2D is as fast or faster at the same error (its error runs about 1e-4 ×
eps), so it stays the default there.

`eps` sets the accuracy of the sum (default `tol`). The FMMs take it per call, `E(f; eps)`; an Ewald plan and the
local sum of a remainder fix it when they are built.

## FMM

`stokes2d_apply` and `stokes3d_plane_apply` wrap FMM2D's and FMM3D's Stokes FMMs in the package's units, for points
in the plane. With ``G_2 = (-\log\rho\,I + rr^T/\rho^2)/(4\pi\mu_{2D})`` the 2D velocity is the library's divided
by ``2\pi\mu_{2D}``, and the stress (and so the traction, `traction_normals`) does not depend on ``\mu_{2D}``; with
``G_3 = (I/\rho + rr^T/\rho^3)/(4\pi\mu_{3D})``, the in-plane block of the half-space Stokeslet, the 3D velocity is
twice the library's divided by ``\mu_{3D}``, and its out-of-plane component vanishes. The tests re-check every
normalization against direct sums (`stokes2d_direct`, `stokes3d_plane_direct`).

FMM3D drops every coincident pair. FMM2D drops the pair ``i = j`` but, for any other coincident pair (an explicit
target on a source, a duplicated source), adds a spurious ``f/(8\pi\mu_{2D})``; `stokes2d_apply` finds such pairs
by hashing and subtracts it.

## Ewald splitting for the 3D Stokeslet

For points in a plane the 3D Stokeslet sums faster by Ewald splitting with non-uniform FFTs (FINUFFT) than by FMM3D.
With ``\phi = 1/\rho`` and ``K[\phi](r) = \phi I - (\phi'(\rho)/\rho)\,rr^T`` (so that ``K[1/\rho] = 4\pi\mu G_3``),
the kernel splits as

```math
\frac{1}{\rho} = \underbrace{\frac{\operatorname{erf}(\rho/\sigma)}{\rho}}_{\phi_L,\ \text{smooth}}
              + \underbrace{\frac{\operatorname{erfc}(\rho/\sigma)}{\rho}}_{\phi_S,\ \text{short range}},
\qquad K = K[\phi_L] + K[\phi_S].
```

- **Long range.** ``K[\phi_L]`` is smooth. Multiplied by a window equal to 1 over every separation in the point set's
  bounding box, it has compact support, so its free-space convolution equals a periodic one on twice the support.
  Its Fourier multiplier is computed once by an FFT of samples; each sum is a type-1 NUFFT (sources to modes), the
  2×2 multiplier, and a type-2 NUFFT (modes to targets).
- **Short range.** ``K[\phi_S]`` decays like ``\operatorname{erfc}(\rho/\sigma)``; the pairs closer than
  ``r_c = c\sigma``, ``\operatorname{erfc}(c) \approx`` `tol`, are summed through a sparse matrix built once.
- **Coincident pairs.** The short range skips them; the long range's value there,
  ``K[\phi_L](0)f = 2f/(\sigma\sqrt\pi)``, is subtracted, for ``i = j`` and for any target exactly on a source.

``\sigma`` is chosen for about `nneigh` (150) short-range neighbors per target. A smooth remainder (a
`SymbolKernel`, below) is added to the long-range multiplier.

With a finite `μ_2D`, `EwaldStokes3D` sums ``G_2 + G_3`` (`StokesKernel`) on one grid, the 2D Stokeslet split as in
`FilmEwald` below; its long-range value at coincident pairs adds ``(1 + \gamma_E/2 - \log\sigma)/(4\pi\mu_{2D})``.
With few targets (a boundary's nodes) the direct sum is faster, and `fast = :auto` takes it (above).

`EwaldLaplace3D` is the same split for the 3D Laplace kernel ``1/(4\pi\rho)``: a scalar multiplier, one NUFFT pair
per sum, and the self value ``\phi_L(0)/4\pi`` subtracted at coincident pairs.

## A whole kernel from its symbol

A `SymbolKernel` used as the kernel itself, of base `Stokeslet3D` or `LaplaceKernel3D`, is summed whole by the
base's Ewald split plus its remainder ``R``, of any decay. ``R`` may have a cone or a logarithm at ``\rho = 0`` (an
algebraically decaying remainder), which a long-range multiplier cannot sample, so it is split by a weight,

```math
R = \underbrace{w R}_{\text{long range}} + \underbrace{(1 - w) R}_{\text{short range}},\qquad
w(\rho) = P(m, U) = 1 - e^{-U}\sum_{j<m} \frac{U^j}{j!},\quad U = \rho^2/\sigma^2,\ m = 4.
```

``w`` is a sum of Gaussians of width ``\sigma`` times polynomials, so the grid of the base's split resolves it, and
``w \sim U^m/m!`` at ``\rho = 0`` cancels a cone of ``R`` to order ``2m``. Its leftover Fourier tail is why an
algebraic remainder takes a grid twice as fine. ``(1 - w) R`` decays like ``e^{-U}U^{m-1}``; the short range runs out
to where it is below `tol`. Coincident pairs are skipped (``w R`` vanishes there), as in the direct sum of the whole
kernel, whose near corrections hold its singular part. The remainder's table (or its far field) must reach the
long-range window, the point set's extent plus about ``11\sigma``; the plan refuses it otherwise.

## Ewald splitting for the 2D kernels

`RadialEwald2D` sums ``\log\rho/2\pi`` and ``K_0(k\rho)/2\pi`` (values, several densities at once, or gradients).
Both are heat-kernel integrals, ``\frac{1}{4\pi}\int_0^\infty e^{-sk^2 - \rho^2/4s}\,ds/s`` up to sign and a constant,
split at ``s = \sigma^2/4``: the long range has the Fourier transform ``e^{-\sigma^2(k^2 + |\xi|^2)/4}/(k^2 + |\xi|^2)``
and the short range decays like ``E_1(\rho^2/\sigma^2)``. For ``\log\rho`` both parts are closed forms (the short range
``-E_1(U)/4\pi``); for ``K_0`` the long range is tabulated once per plan on ``\rho \le \sqrt{40}\,\sigma`` and equals
``K_0`` beyond. A gradient multiplies the multiplier by ``i\xi``. `BiharmonicGradient2D` is three log sums and moments,
as on its FMM path.

With a remainder (a finite depth: symbol ``\Phi(kH)/k^4`` over the deep ``1/k^4``), `out = :bgrad` sums the whole
kernel ``r f(\rho)``, ``f = (2\log\rho + c_0)/8\pi + q(\rho)``, with ``q`` the remainder's regularized `:gradient`
and ``c_0`` the deep kernel's, which together make ``f`` decay. The same split of ``\log\rho`` gives the short range
``-r E_1(U)/8\pi`` and the long range ``r[(2\log_L\rho + c_0)/8\pi + q]``, sampled as a two-component kernel, so the
deep part and ``q`` cancel when the multiplier is built rather than in each apply (the moments of the deep path would
lose about ``L\log L/H`` in relative accuracy). An apply is one type-1 NUFFT, a 2 × 1 multiplier and a type-2 NUFFT
with two outputs. Both parts vanish at ``r = 0``. The grid resolves ``q`` (``\sigma \le cH/10``), and the remainder's
table or its far field must reach the long-range window. The plan checks that ``f`` decays at the table's end, which
catches a ``c_0`` (a ``\kappa``) that is not the remainder's. On a film 9 across at ``H = 2``, eps 1e-7, 72k nodes,
an apply takes 0.2 s against 32 s for the local sum, which is all-pairs there (``\rho_c = 20.5``).

`FilmEwald` evaluates, on one grid, all the far-field sums of a coupled 2D + 3D Stokes operator on a surface and its
edge: the area-to-area velocity ``\sum_j [G_2 + G_3](x_i - y_j)f_j``, the edge-to-area velocity
``\sum_j G_2(x_i - z_j)g_j``, and the area-to-edge traction of the 2D part. The 2D Stokeslet is split by Hasimoto's
screening of the biharmonic (``U = \rho^2/\sigma^2``),
``-\log\rho\,I + rr^T/\rho^2 = S_L + S_S``, ``S_S = (\tfrac12 E_1(U) - e^{-U})I + e^{-U}rr^T/\rho^2``.

## Smooth convolutions

`SmoothConvolution` sums a kernel that is smooth on a given length scale (the remainder of a kernel from its
symbol) over all pairs, including coincident ones, by the long-range half of the Ewald method alone.

## Threads

The FMMs, the NUFFTs and the corrections run on Julia's threads. FINUFFT and FFTW.jl share a libfftw3 whose threads
callback (installed by FFTW.jl) FINUFFT's threaded FFTs cannot run under; the package removes the callback when it
loads with more than one thread, which leaves FFTW its own threads (the setting is process-wide).
