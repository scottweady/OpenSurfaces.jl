# Volume potentials (src/evaluators/volume.jl) on the unit disc against the closed forms of Gomez
# Herrera & Weady (2026), appendix F: V_D[ζ^m] = (|ζ|²/(4(m+1)) + b_m) ζ^m and B_D[ζ^m] = (a_m|ζ|⁴ + b_m|ζ|² + c_m) ζ^m
# (κ = 0), for the real densities Re ζ^m, m = 0–3: ∇V and ∇B at the area nodes and on Γ.
#   julia -t 8 --project -e 'using Pkg; Pkg.test(test_args=["volume"])'   (from the package root)
using OpenSurfaces, LinearAlgebra, Printf, Test

Vc(m) = (1 / (4(m + 1)), m == 0 ? 0.0 : -1 / (4m))
Bc(m) = (1 / (32(m + 1) * (m + 2)), m == 0 ? -1 / 16 : -1 / (16m * (m + 1)),
         m == 0 ? -5 / 64 : m == 1 ? 3 / 64 : 1 / (32m * (m - 1)))
"""∇ Re[(a|ζ|⁴ + b|ζ|² + c) ζ^m] at x."""
function gradpoly(x, m, a, b, c)
    ζ = complex(x[1], x[2]); r2 = abs2(ζ)
    h = a * r2^2 + b * r2 + c; dh = 4a * r2 + 2b          # ∂ₓh = dh·x, ∂ᵧh = dh·y
    zm = ζ^m; dz = m == 0 ? 0.0im : m * ζ^(m - 1)
    Vec2(real(dh * x[1] * zm + h * dz), real(dh * x[2] * zm + h * im * dz))
end

@testset "∇V and ∇B on the unit disc against appendix F" begin
    p, tol = 10, 1e-6
    bd, ad, _ = build_mesh(discretize_curve(θ -> Vec2(cos(θ), sin(θ)), p; tol), p; edge_layers=2)
    VP = VolumePotentials(ad, bd; tol, eps=tol)
    ev, eb = 0.0, 0.0
    for m in 0:3
        g = [real(complex(x[1], x[2])^m) for x in ad.x]
        gv = volume_grad(VP, g); gb = volume_bgrad(VP, g)
        a, b = Vc(m); A, B, C = Bc(m)
        rv(x) = gradpoly(x, m, 0.0, a, b); rb(x) = gradpoly(x, m, A, B, C)
        sv = maximum(norm.(rv.(bd.x))); sb = maximum(norm.(rb.(bd.x)))
        ev = max(ev, maximum(norm.(gv.area .- rv.(ad.x))) / sv, maximum(norm.(gv.bdry .- rv.(bd.x))) / sv)
        eb = max(eb, maximum(norm.(gb.area .- rb.(ad.x))) / sb, maximum(norm.(gb.bdry .- rb.(bd.x))) / sb)
    end
    @printf("volume potentials on the disc, m = 0–3: ∇V %.1e, ∇B %.1e (relative to the maximum on Γ)\n", ev, eb)
    @test ev < 10tol && eb < 10tol
end

@testset "∇B at finite depth: Ewald of the whole kernel against the local sum" begin
    # the bilaplacian over a layer of depth H, symbol Φ(kH)/k⁴ (studies/bgrad_ewald/TODO.md): the deep kernel at
    # c0 = −1 + 2K0 − 2 log 2H plus the regularized remainder (Φ − 1)/k⁴, its far field −(2 log ρ + c0)/8π
    function Φ(x)
        x < 0.1 && return x^4 * evalpoly(x^2, (1 / 10, -163 / 3150, 49 / 2250, -80614 / 9095625, 19115233 / 5320940625))
        x > 30 && return 1.0
        t = tanh(x); ic2 = 1 / cosh(x)^2
        (t^3 - x^3 * ic2) / (t - x * ic2)
    end
    K0 = 1.3476268494                     # γ_E − ∫₀^∞ (Φ(s) − 1_{s>1}) ds/s
    H, tol = 0.5, 1e-7; c0 = -1 + 2K0 - 2log(2H)
    rem(far) = SymbolKernel(k -> Φ(k * H) / k^4; base=LaplaceKernel3D(), scale=H, outputs=(:gradient,), regularize=true,
                            remainder=k -> (Φ(k * H) - 1) / k^4, q=24, nq=20,
                            far=far ? (ρ -> (-(2log(ρ) + c0) / 8π,)) : nothing)
    p = 10
    bd, ad, _ = build_mesh(discretize_curve(θ -> Vec2(1.5cos(θ), 0.8sin(θ)), p; tol), p; edge_layers=2)
    ev(R; kw...) = Evaluator(BiharmonicGradient2D(c0), ad; bd, density=:smooth, targets=(:area, :boundary), remainder=R,
                             tol, eps=tol, kw...)
    EE = ev(rem(true)); EL = ev(rem(false))
    @test EE.far isa OpenSurfaces._RadialEwaldFar && EL.far isa OpenSurfaces._LocalFar
    σ = [1 + 0.3sin(2x[1]) * cos(x[2]) for x in ad.x]
    ue = EE(σ); ul = EL(σ)
    e = max(maximum(norm.(ue[1] .- ul[1])) / maximum(norm.(ul[1])), maximum(norm.(ue[2] .- ul[2])) / maximum(norm.(ul[2])))
    @printf("∇B at depth H = %g: Ewald of the whole kernel against the local sum %.1e\n", H, e)
    @test e < tol
    # a deep kernel whose c0 is not the remainder's leaves f ~ log ρ: refused
    @test_throws ArgumentError Evaluator(BiharmonicGradient2D(; κ=0.3), ad; bd, density=:smooth, remainder=rem(true),
                                         targets=(:area, :boundary), fast=:ewald)
end
