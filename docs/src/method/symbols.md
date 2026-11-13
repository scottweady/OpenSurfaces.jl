# [Kernels from Fourier symbols](@id method-symbols)

## A standard kernel plus a smooth remainder

Many kernels of interest are known by their Fourier transform rather than in closed form: the potential of a layer
of finite depth, a screened kernel, the response of a membrane embedded in a fluid. For a radial kernel with symbol
``\hat s(k)``, ``k = |\mathbf k|`` (or a tensor kernel with isotropic mobilities,
``\hat G = a(k)(I - \hat k\hat k^T) + b(k)\hat k\hat k^T``), `SymbolKernel` writes the symbol as that of a standard
kernel `base` with the same large-``k`` behavior plus a remainder,

```math
\hat s(k) = \hat s_{\text{base}}(k) + \Delta(k).
```

``\Delta`` decays in ``k``, so its kernel ``R`` is smooth in real space, and is tabulated once by Hankel transforms
(``x = k\rho``):

| Output | |
|:--|:--|
| `:value` | ``R(\rho) = \frac{1}{2\pi}\int_0^\infty \Delta\,J_0(x)\,k\,dk`` |
| `:gradient` | ``R'(\rho)/\rho = -\frac{1}{2\pi}\int_0^\infty \Delta\,k^2\,\frac{J_1(x)}{x}\,k\,dk`` (``\nabla R = (R'/\rho)\,r``) |
| `:potential_gradient` | ``P'(\rho)/\rho = \frac{1}{2\pi}\int_0^\infty \Delta\,\frac{J_1(x)}{x}\,k\,dk`` (``\Delta P = R``) |
| `:potential` | ``P(\rho) - P(0) = \frac{1}{2\pi}\int_0^\infty \Delta\,\frac{1 - J_0(x)}{k^2}\,k\,dk`` |
| tensor | ``R = \alpha I + \beta rr^T``, ``\alpha = A_0 + A_2``, ``\beta = -2A_2/\rho^2``, ``A_0 = \frac{1}{2\pi}\int s\,J_0\,k\,dk``, ``A_2 = \frac{1}{2\pi}\int d\,J_2\,k\,dk``, ``s = (\Delta a + \Delta b)/2``, ``d = (\Delta b - \Delta a)/2`` |

The ``k`` integrals are Gauss–Legendre on panels (uniform on ``[0, k_{\max}]``, or on given breakpoints when
``\Delta`` has structure near ``k = 0``); the table is piecewise Chebyshev on panels of width about `scale` up to
``\rho_c`` (`RadialTable`), with a closed-form far field beyond it if one is given.

The numerics keep the standard kernel for everything singular (the local corrections, Ewald's short range, the
self terms, preconditioners) and add only the smooth remainder: to Ewald's long-range multiplier (`Stokeslet3D`), or
as a local sum over the pairs closer than ``\rho_c`` where the whole kernel decays (`LaplaceKernel3D`,
`BiharmonicGradient2D`), or through `SmoothConvolution`.

```julia
H = 0.3     # the 3D Laplace kernel over a layer of depth H, the value fixed on its floor: tanh(kH)/(2k)
K = SymbolKernel(k -> tanh(k * H) / (2k); base = LaplaceKernel3D(), scale = H, outputs = (:value,),
                 remainder = k -> -1 / (k * (exp(2k * H) + 1)))
K(Vec2(0.1, 0.2))             # the whole kernel, 1/(4πρ) + R(ρ)
kernel_remainder(K, 0.5)      # R alone
E = Evaluator(LaplaceKernel3D(), ad; remainder = K)
```

## Conditions on the base and the remainder

- **The base** must match the symbol at large ``k`` (the same singularity, so that ``\Delta`` decays and ``R`` is
  smooth) and at small ``k`` (``k\,\Delta`` integrable at ``k = 0``, so the Hankel integrals converge; for a gradient
  output only ``k^3\Delta``). It is a standard kernel (`Stokeslet3D`, `LaplaceKernel3D`) or one with a
  `kernel_symbol` method. It is given, not extracted from the symbol. The constructor checks the small-``k`` limit
  and throws when the integrals diverge.
- **`remainder`** gives ``\Delta`` in a numerically stable form when the plain difference cancels (as for the finite
  depth above, whose difference is ``-1/(k(e^{2kH} + 1))``).
- **A remainder not integrable at ``k = 0``** (``\Delta \sim k^{-4}``) has a gradient only up to a multiple of
  ``r``: `regularize = true` subtracts the ``x \to 0`` value of ``J_1(x)/x`` (`:gradient` output only), and the base
  must fix that multiple.
- **How the remainder decays.** A remainder that decays exponentially in ``k`` (a finite depth) is resolved by the
  default ``k_{\max} = 25/``scale. One that decays algebraically (a screened kernel ``e^{-\kappa\rho}/4\pi\rho``,
  ``\Delta \sim -\kappa^2/(4k^3)``) has a cone or a logarithm at ``\rho = 0``: the constructor classifies the tail
  (`K.decay === :algebraic`), extends ``k_{\max}`` (up to 64 times) until the tail is below `tol` of ``|R(0)|``,
  and warns when it cannot. An `Evaluator` then takes the near corrections of the whole kernel, base plus remainder,
  and the Ewald sums refuse it.

`cached_kernel(build, key)` keeps a table built once per key (a depth, a screening length) for the session.
