# Volume potentials on the film for growth and buoyancy (Gomez Herrera & Weady 2026). For densities given at the area nodes (physical values):
#   V_Ω[g](x)      = (1/2π) ∫_Ω log|x − y| g(y) dy             (ΔV_Ω[g] = g in Ω; `potential = true`),
#   ∇V_Ω[g](x)     = (1/2π) ∫_Ω (x − y)/|x − y|² g(y) dy,     Φ = V_Ω[g] has ΔΦ = g in Ω,
#   ∇B_{Ω,κ}[σ](x) = (1/8π) ∫_Ω (x − y) (2 log(|x − y|/R) − 1 + 2κ) σ(y) dy,   R = (|Ω|/π)^{1/2},
# at the area nodes and at the Γ nodes. B is the paper's bilaplacian volume potential
# (Δ²B = σ); −(Ra/16)∇B[σ] is the surface velocity of the buoyant bulk with a stress-free surface.
# Each potential is an `Evaluator` (src/evaluators/evaluator.jl) of the kernels LaplaceKernel2D,
# LaplaceGradient2D, BiharmonicGradient2D: FMM2D (Laplace) with the smooth rule, sparse near corrections from
# `area_correction`. The densities are smooth, so the sources are an unweighted copy of the mesh (Gauss–Legendre in
# t on every panel; on the edge-weighted panels g is interpolated from the Gauss–Jacobi nodes). Integrating g √(1+t)
# with the weighted rule instead converges only algebraically (5e-5 in ∇V on the unit disc).
# A `remainder` (a scalar `SymbolKernel` with the regularized :gradient, e.g. a finite-depth buoyancy
# kernel K_H): ∇B is replaced by the deep kernel at κ (given, or a function of R) plus the
# smooth q(ρ) r. The whole kernel decays exponentially (an FMM of the deep part alone would cancel against the
# remainder): with a far field on the remainder it is summed by Ewald (`RadialEwald2D`, out = :bgrad, the whole kernel
# in the long-range multiplier); without one, directly over the pairs closer than ρ_c, where the envelope of ρ|∇K|
# falls below eps of its maximum (all pairs when ρ_c exceeds the film). The near corrections are those of the deep
# kernel (the remainder is smooth).

struct VolumePotentials
    ad::AreaDisc                        # targets (the film mesh)
    bd::BoundaryDisc
    R::Float64                          # R(Ω) = (|Ω|/π)^{1/2}
    κ::Float64                          # of the deep bilaplacian kernel (with a remainder: its κ at R)
    EV::Union{Nothing,Evaluator}        # ∇V at (area, Γ)
    E0::Union{Nothing,Evaluator}        # V at the area and Γ nodes (`volume_potential` returns the area values)
    EB::Union{Nothing,Evaluator}        # ∇B at (area, Γ)
end

"""
    VolumePotentials(ad, bd; κ=0, remainder=nothing, tol=DEFAULT_TOL, eps=tol, grad=true, buoyancy=true,
                     potential=false)

∇V_Ω (with `grad`) and ∇B_{Ω,κ} (with `buoyancy`) at the area and Γ nodes, and (with `potential`) V_Ω itself at the
area nodes. A `remainder` (scalar `SymbolKernel`, regularized :gradient) makes `volume_bgrad` the deep kernel at κ
plus the remainder, summed by Ewald when the remainder has a far field and locally otherwise (all pairs when the
kernel's range exceeds the film); `κ` may then be a function of R(Ω). `tol` for the near corrections, `eps` for the
FMM (with a remainder: the Ewald plan, or the relative size of the truncated kernel).
"""
function VolumePotentials(ad::AreaDisc, bd::BoundaryDisc; κ=0.0, remainder=nothing, tol::Real=DEFAULT_TOL, eps::Real=tol,
                          grad::Bool=true, buoyancy::Bool=true, potential::Bool=false)
    area = sum(dot(x, n) * w for (x, n, w) in zip(bd.x, bd.n, bd.w)) / 2
    R = sqrt(area / π)
    κ = Float64(κ isa Function ? κ(R) : κ)
    # one correction pass for all the kernels asked for (`Evaluator((K1, K2, …), ad)`), at the area and Γ nodes (V's
    # Γ values are not used: they cost little against a second pass)
    ks = Any[]; rs = Any[]
    (grad || potential) && (push!(ks, LaplaceGradient2D()); push!(rs, nothing))
    potential && (push!(ks, LaplaceKernel2D()); push!(rs, nothing))
    buoyancy && (push!(ks, BiharmonicGradient2D(; κ, R)); push!(rs, remainder))
    es = isempty(ks) ? () : Evaluator(Tuple(ks), ad; bd=bd, density=:smooth, tol=tol, eps=eps, targets=(:area, :boundary),
                                      remainder=Tuple(rs))
    i = 0; take(on) = on ? es[i+=1] : nothing
    VolumePotentials(ad, bd, R, κ, take(grad || potential), take(potential), take(buoyancy))
end

"""∇V_Ω[g] at the area nodes and at Γ, for physical nodal values g: `(area, bdry)`."""
function volume_grad(VP::VolumePotentials, g::AbstractVector{<:Real})
    VP.EV === nothing && error("volume_grad: construct the VolumePotentials with grad=true")
    ua, ub = VP.EV(g)
    (area=ua, bdry=ub)
end

"""V_Ω[g] at the area nodes and ∇V_Ω[g] at the area nodes and at Γ, for physical nodal values g:
`(V, area, bdry)` (needs `VolumePotentials(…; potential=true)`)."""
function volume_potential(VP::VolumePotentials, g::AbstractVector{<:Real})
    VP.E0 === nothing && error("volume_potential: construct the VolumePotentials with potential=true")
    ua, ub = VP.EV(g)
    (V=VP.E0(g)[1], area=ua, bdry=ub)                    # E0 at (area, Γ): the area values
end

"""∇B_{Ω,κ}[σ] at the area nodes and at Γ, for physical nodal values σ: `(area, bdry)`."""
function volume_bgrad(VP::VolumePotentials, σ::AbstractVector{<:Real})
    VP.EB === nothing && error("volume_bgrad: construct the VolumePotentials with buoyancy=true")
    ua, ub = VP.EB(σ)
    (area=ua, bdry=ub)
end
