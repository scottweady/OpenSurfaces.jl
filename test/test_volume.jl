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
