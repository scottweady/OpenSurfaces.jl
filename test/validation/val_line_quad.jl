# Validation for src/quadrature/line_quad.jl: error vs p and K on a star (printed convergence report).
#   run by run_all.jl (from the package root; `push!(ARGS, "line_quad")` before its include runs this file only)

include(joinpath(@__DIR__, "..", "common", "line_quad_common.jl"))

# ---------------------------------------------------------------------------------------
# Convergence report (printed, not tested)
# ---------------------------------------------------------------------------------------
println("\nerror vs p and K (star r = 1 + 0.3 cos 5θ), smooth ν; quadrature error only:")
println("  p    K   on-Γ velocity    near velocity    T_ΓΓ vs PV ref")
for p in (8, 10, 12, 16), K in (32, 64, 128)
    bd = curve_disc(star(), K, p); lq = LineQuad_(bd; tol=1e-10); ν = smooth_density(bd)
    u = vecs(slp2d_velocity_matrix(lq) * flat(ν))
    e_on = maximum(norm(u[it] - ref_total_node(G2t, bd, it, ν)) for it in (1, p ÷ 2, 5p + 3))
    tg = near_targets(bd, lq, (1,), (-0.999, 0.3), (1e-2, 1e-8))
    un = slp2d_eval(lq, tg, ν)
    e_near = maximum(norm(un[m] - ref_total(G2t, bd, tg[m], ν)) for m in eachindex(tg))
    t = vecs(traction_self_matrix(bd) * flat(ν))
    e_t = maximum(norm(t[it] - 0.5ν[it] - ref_traction_pv(bd, it, ν)) for it in (1, 5p + 3))
    println("  ", lpad(p, 2), "  ", lpad(K, 3), "   ", rpad(round(e_on, sigdigits=2), 15), "  ",
            rpad(round(e_near, sigdigits=2), 15), "  ", round(e_t, sigdigits=2))
end
