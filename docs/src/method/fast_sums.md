# [Fast sums](@id method-fast-sums)

The plain rule is a sum over all (target, source) pairs, ``u_i = \sum_j K(x_i - y_j)\,f_j`` with the weights in
``f``. Every fast sum here skips coincident pairs (a target exactly on a source), as the local corrections expect.
An `Evaluator` picks the sum by the kernel's type:

| Kernel | Sum |
|:--|:--|
| `Stokeslet3D` | Ewald + NUFFT (`EwaldStokes3D`; default), or FMM3D (`fast = :fmm`) |
| `LaplaceKernel3D` | FMM3D (Laplace) |
| `LaplaceKernel2D`, `LaplaceGradient2D`, `BiharmonicGradient2D` | FMM2D (Laplace) |
| `MHKernel`, `MHNormalKernel` | FMM2D (Helmholtz at the imaginary wavenumber ``ik``) |
| any other | direct |

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
