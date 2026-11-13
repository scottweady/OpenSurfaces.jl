# Boundary curve discretization.
#
#   discretize_curve(γ, p; ...)    adaptive CurvePanels from a 2π-periodic parametrization
#   local_feature_size(panels; ...) curvature / thickness based local length scale
#   curve_length, curve_area, panel_length, panel_tail   small geometric utilities
#
# A CurvePanel with p Legendre coefficients is the degree p−1 interpolant of the curve at the p
# Gauss–Legendre nodes of its parameter interval. Resolution is measured by the "tail"
# (norm of the last two coefficients) relative to the panel length.

using StaticArrays, LinearAlgebra

# ---------------------------------------------------------------------------------------
# Cached Legendre transforms
# ---------------------------------------------------------------------------------------

const _LEG_CACHE = Dict{Int,Tuple{Vector{Float64},Vector{Float64},Matrix{Float64}}}()
const _LEG_LOCK = ReentrantLock()

"""(x, w, F): n-point Gauss–Legendre nodes/weights and the matrix F mapping nodal values
to Legendre coefficients (c = F v)."""
function _leg_data(n::Int)
    lock(_LEG_LOCK) do
        get!(_LEG_CACHE, n) do
            x, w = gausslegendre(n)
            F = [(2k + 1) / 2 * w[i] * legendre_P(k, x[i]) for k in 0:n-1, i in 1:n]
            (x, w, F)
        end
    end
end

function _apply_F(F::Matrix{Float64}, v::AbstractVector{Vec2})
    n = size(F, 1)
    c = Vector{Vec2}(undef, n)
    @inbounds for k in 1:n
        acc = zero(Vec2)
        for i in 1:n
            acc += F[k, i] * v[i]
        end
        c[k] = acc
    end
    c
end

"""Legendre coefficients (length n) of the interpolant of `f` (s ∈ [-1,1] ↦ Vec2) at n GL nodes."""
function _fit(f, n::Int)
    x, _, F = _leg_data(n)
    _apply_F(F, [Vec2(f(s)) for s in x])
end

_cross(a, b) = a[1] * b[2] - a[2] * b[1]

# ---------------------------------------------------------------------------------------
# Simple geometric utilities
# ---------------------------------------------------------------------------------------

"""Arclength of the polynomial curve with Legendre coefficients `c` (GL quadrature)."""
function panel_length(c::AbstractVector{Vec2}; nq::Int = 2length(c) + 2)
    x, w, _ = _leg_data(nq)
    L = 0.0
    for i in eachindex(x)
        L += w[i] * norm(legendre_eval(c, x[i])[2])
    end
    L
end
panel_length(P::CurvePanel; kw...) = panel_length(P.c; kw...)

"""Resolution measure: norm of the last two Legendre coefficients (absolute)."""
function panel_tail(c::AbstractVector{Vec2})
    n = length(c)
    n <= 2 && return 0.0
    sqrt(sum(abs2, c[n]) + sum(abs2, c[n-1]))
end
panel_tail(P::CurvePanel) = panel_tail(P.c)

curve_length(panels::AbstractVector{CurvePanel}) = sum(panel_length, panels)

"""Area enclosed by the (closed, counterclockwise) panel curve: ½∮(x dy − y dx), exact up
to rounding for polynomial panels."""
function curve_area(panels::AbstractVector{CurvePanel})
    A = 0.0
    for P in panels
        nq = 2length(P.c)
        x, w, _ = _leg_data(nq)
        for i in eachindex(x)
            g, dg = curve_eval(P, x[i])
            A += 0.5 * w[i] * _cross(g, dg)
        end
    end
    A
end

"""Signed area of a closed polygon (vertices not repeated)."""
function polygon_area(v::AbstractVector)
    n = length(v); A = 0.0
    for i in 1:n
        a = v[i]; b = v[mod1(i + 1, n)]
        A += a[1] * b[2] - a[2] * b[1]
    end
    A / 2
end

"""Is point x inside the closed polygon `v` (even–odd rule)?"""
function point_in_polygon(x, v::AbstractVector)
    n = length(v); inside = false
    j = n
    for i in 1:n
        a = v[i]; b = v[j]
        if (a[2] > x[2]) != (b[2] > x[2])
            xc = a[1] + (x[2] - a[2]) * (b[1] - a[1]) / (b[2] - a[2])
            x[1] < xc && (inside = !inside)
        end
        j = i
    end
    inside
end

"""Reverse the orientation of a closed panel curve (panel order reversed, s → −s on each panel)."""
reverse_curve(panels::AbstractVector{CurvePanel}) =
    [CurvePanel([(isodd(i) ? 1 : -1) * c for (i, c) in enumerate(P.c)]) for P in reverse(panels)]

"""Polyline approximation of the panel curve, `m` segments per panel (equispaced in s)."""
function curve_polyline(panels::AbstractVector{CurvePanel}, m::Int)
    pts = Vec2[]
    for P in panels, j in 0:m-1
        push!(pts, curve_eval(P, -1 + 2j / m)[1])
    end
    pts
end

# ---------------------------------------------------------------------------------------
# Uniform bucket grid (neighbor search without extra packages)
# ---------------------------------------------------------------------------------------

struct _BucketGrid
    h::Float64
    cells::Dict{Tuple{Int,Int},Vector{Int}}
end

_cellidx(g::_BucketGrid, x) = (floor(Int, x[1] / g.h), floor(Int, x[2] / g.h))

function _BucketGrid(pts::AbstractVector, h::Float64)
    g = _BucketGrid(h, Dict{Tuple{Int,Int},Vector{Int}}())
    for (i, x) in enumerate(pts)
        push!(get!(Vector{Int}, g.cells, _cellidx(g, x)), i)
    end
    g
end

"""Insert item `i` covering the box [lo, hi] into all overlapped cells."""
function _insert_box!(g::_BucketGrid, i::Int, lo, hi)
    a = _cellidx(g, lo); b = _cellidx(g, hi)
    for cx in a[1]:b[1], cy in a[2]:b[2]
        push!(get!(Vector{Int}, g.cells, (cx, cy)), i)
    end
end

"""Call f(i) for every item in cells overlapping the box [lo, hi] (items may repeat)."""
function _foreach_in_box(f, g::_BucketGrid, lo, hi)
    a = _cellidx(g, lo); b = _cellidx(g, hi)
    ncell = (b[1] - a[1] + 1) * (b[2] - a[2] + 1)
    if ncell > 4 * length(g.cells)   # huge box: just visit every occupied cell
        for (key, items) in g.cells
            (a[1] <= key[1] <= b[1] && a[2] <= key[2] <= b[2]) || continue
            foreach(f, items)
        end
    else
        for cx in a[1]:b[1], cy in a[2]:b[2]
            items = get(g.cells, (cx, cy), nothing)
            items === nothing || foreach(f, items)
        end
    end
end

# ---------------------------------------------------------------------------------------
# Local feature size
# ---------------------------------------------------------------------------------------

"""
    local_feature_size(panels; radius_factor=4.0, cosmax=0.0, nsamp=length(c))

Estimate of the local feature size of Γ at each panel, combining curvature and thickness:

    lfs(x) = min( 1/|κ(x)|,  ½ min{ |x − y| : y ∈ Γ sample, n(x)·n(y) < cosmax } ).

The second term is a medial-axis style "half distance to the nearest part of Γ facing away
from x": across a neck of half-width δ it gives ≈ δ; it also responds to near-contact of two
parts of Γ from the outside. The search for y is limited to radius `radius_factor × L` (L
the length of the panel containing x, the larger neighbor at a vertex); if nothing is found
the thickness term is taken as Inf (callers choose the radius large enough that larger
values would not matter).

Γ is sampled at each panel's start point and `nsamp` Gauss–Legendre points.
Returns `(vertex, panel, vertex_interior)`: lfs at the start vertex of each panel, the minimum
over each panel (including both endpoints), and the interior-only variant at the vertices,
min(1/κ₊, ½ distance to points y facing x through Ω: (y−x)·n(x) < 0 < (y−x)·n(y)), with κ₊
the convex part of the curvature. The interior variant is what limits the strip width (near
contact from the outside and concave curvature do not), while the full lfs is used for panel
lengths (quadrature accuracy).
"""
function local_feature_size(panels::AbstractVector{CurvePanel}; radius_factor::Real = 4.0,
                            cosmax::Real = 0.0, nsamp::Int = length(panels[1].c),
                            cstart::AbstractVector{Int} = [1, length(panels) + 1])
    N = length(panels)
    nxt, prv = cyclic_maps(cstart)
    xs, _, _ = _leg_data(nsamp)
    svals = [-1.0; xs]
    m = length(svals)
    X = Vector{Vec2}(undef, N * m); Nn = similar(X); K = zeros(N * m); R = zeros(N * m)
    L = [panel_length(P) for P in panels]
    for (k, P) in enumerate(panels)
        dc = legendre_deriv_coefs(P.c)
        for (j, s) in enumerate(svals)
            idx = (k - 1) * m + j
            g, dg = curve_eval(P, s)
            _, dd = legendre_eval(dc, s)
            sp = norm(dg); τ = dg / sp
            X[idx] = g; Nn[idx] = Vec2(τ[2], -τ[1])
            K[idx] = _cross(dg, dd) / sp^3
            # search radius: local panel length (for the vertex, the smaller neighbor)
            Lloc = j == 1 ? max(L[k], L[prv[k]]) : L[k]
            R[idx] = radius_factor * Lloc
        end
    end
    h = sort(L)[cld(N, 2)] / 2
    grid = _BucketGrid(X, h)
    lfs = similar(K); lfsi = similar(K)
    for i in eachindex(X)
        x = X[i]; n = Nn[i]
        r = R[i]
        best = Inf; besti = Inf
        _foreach_in_box(grid, x .- r, x .+ r) do j
            if dot(n, Nn[j]) < cosmax
                d = X[j] - x
                d2 = sum(abs2, d)
                if d2 <= r^2
                    d2 < best && (best = d2)
                    # facing through the interior: y lies inward of x and x inward of y
                    (dot(d, n) < 0 && dot(d, Nn[j]) > 0 && d2 < besti) && (besti = d2)
                end
            end
        end
        lfs[i] = min(sqrt(best) / 2, 1 / max(abs(K[i]), 1e-300))
        lfsi[i] = min(sqrt(besti) / 2, 1 / max(K[i], 1e-300))
    end
    vert = [lfs[(k-1)*m+1] for k in 1:N]
    pan = [min(minimum(@view lfs[(k-1)*m+1:k*m]), vert[nxt[k]]) for k in 1:N]
    verti = [lfsi[(k-1)*m+1] for k in 1:N]
    (vertex = vert, panel = pan, vertex_interior = verti)
end

# ---------------------------------------------------------------------------------------
# Adaptive discretization of a parametrized curve
# ---------------------------------------------------------------------------------------

"""
    discretize_curve(γ, p; tol=DEFAULT_TOL, maxlen=Inf, lfs_factor=1.0, nstart=8,
                     balance=true, maxpanels=100_000, return_params=false, lencap=nothing)

Adaptive panel discretization of the closed curve θ ↦ γ(θ) (2π-periodic, counterclockwise,
returning anything convertible to Vec2). Only γ itself is needed: each panel is the degree
p−1 interpolant at p Gauss–Legendre nodes in θ, and is bisected (in θ) until

  * resolution: norm of the last two Legendre coefficients ≤ max(tol × panel length,
    1000 eps × curve size) (the second term is a rounding-error floor),
  * panel length ≤ maxlen, and ≤ lencap(x) at the panel midpoint x when a function `lencap` is given,
  * panel length ≤ lfs_factor × local_feature_size (thin necks, near-contact, high curvature;
    lfs_factor = Inf disables),
  * balance: neighboring panel lengths differ by at most a factor 2 (if `balance`).

Returns `Vector{CurvePanel}` (each with p coefficients), or `(panels, θbreaks)` when
`return_params=true` (panel k covers θbreaks[k] .. θbreaks[k+1]).
"""
function discretize_curve(γ, p::Int; tol::Real = DEFAULT_TOL, maxlen::Real = Inf, lfs_factor::Real = 1.0,
                          nstart::Int = 8, balance::Bool = true, maxpanels::Int = 100_000,
                          return_params::Bool = false, lfs_cosmax::Real = 0.0, lencap = nothing)
    fit(a, b) = _fit(s -> γ(a + (b - a) * (s + 1) / 2), p)
    ab = Tuple{Float64,Float64}[]
    cs = Vector{Vec2}[]
    Ls = Float64[]
    minwidth = 2π * 2.0^-36
    unresolved = Float64[]
    # rounding-error floor for the tail test (γ is only known to ~eps × its size; evaluations
    # with cancellation are worse)
    scale = maximum(norm(Vec2(γ(θ))) for θ in range(0, 2π, length = 4nstart + 1))
    floor_abs = 1000 * eps() * scale

    function addresolved!(a, b, c)
        L = panel_length(c)
        long = L > maxlen || (lencap !== nothing && L > lencap(Vec2(γ((a + b) / 2))))
        if (panel_tail(c) > max(tol * L, floor_abs) || long) && (b - a) > minwidth
            length(ab) > maxpanels && error("discretize_curve: more than $maxpanels panels")
            m = (a + b) / 2
            addresolved!(a, m, fit(a, m)); addresolved!(m, b, fit(m, b))
        else
            (b - a) <= minwidth && push!(unresolved, a)
            push!(ab, (a, b)); push!(cs, c); push!(Ls, L)
        end
    end
    θ0 = range(0, 2π, length = nstart + 1)
    for k in 1:nstart
        addresolved!(θ0[k], θ0[k+1], fit(θ0[k], θ0[k+1]))
    end

    # rebuild the panel list, bisecting those marked
    # (returns true if anything was bisected; panels narrower than minwidth are left alone)
    function bisect_marked!(mark)
        abo, cso, Lso = copy(ab), copy(cs), copy(Ls)
        empty!(ab); empty!(cs); empty!(Ls)
        any_split = false
        for k in eachindex(abo)
            if mark[k] && abo[k][2] - abo[k][1] > minwidth
                any_split = true
                a, b = abo[k]; m = (a + b) / 2
                addresolved!(a, m, fit(a, m)); addresolved!(m, b, fit(m, b))
            else
                push!(ab, abo[k]); push!(cs, cso[k]); push!(Ls, Lso[k])
            end
        end
        length(ab) > maxpanels && error("discretize_curve: more than $maxpanels panels")
        any_split
    end

    for _ in 1:200
        changed = false
        if isfinite(lfs_factor)
            panels = [CurvePanel(c) for c in cs]
            lfs = local_feature_size(panels; radius_factor = 3.0 / min(lfs_factor, 1.0),
                                     cosmax = lfs_cosmax).panel
            mark = Ls .> lfs_factor .* lfs
            any(mark) && bisect_marked!(mark) && (changed = true)
        end
        if balance
            for _ in 1:100
                N = length(Ls)
                mark = [Ls[k] > 2 * min(Ls[mod1(k - 1, N)], Ls[mod1(k + 1, N)]) for k in 1:N]
                (any(mark) && bisect_marked!(mark)) || break
                changed = true
            end
        end
        changed || break
    end
    isempty(unresolved) || @warn "discretize_curve: $(length(unresolved)) panel(s) could not be resolved " *
        "to tol (curve not smooth?), near θ = $(unique(round.(unresolved, digits = 6))[1:min(end, 5)])"
    panels = [CurvePanel(c) for c in cs]
    if return_params
        return panels, [[a for (a, _) in ab]; ab[end][2]]
    end
    panels
end

"""
    discretize_curves(γs, p; gap_factor=4.0, return_params=false, kw...)

The curves of one film or plate (outline and holes) as by `discretize_curve`, each also limited
to panels no longer than `gap_factor` × the distance from the panel midpoint to the other curves.
`discretize_curve` sees one curve at a time, so a hole close to the rim (a narrow ring) would
otherwise keep panels much longer than the gap, which the strip mesher cannot fit. Curves that
are well separated are discretized exactly as by `discretize_curve`. Returns one result of
`discretize_curve` per curve.
"""
function discretize_curves(γs::AbstractVector, p::Int; gap_factor::Real = 4.0, return_params::Bool = false, kw...)
    out = [discretize_curve(γ, p; return_params = true, kw...) for γ in γs]
    if length(γs) > 1
        samples(ps) = [curve_eval(P, s)[1] for P in ps for s in range(-1, 1, length = 2length(P.c))]
        for _ in 1:6
            pts = [samples(first(o)) for o in out]
            changed = false
            for i in eachindex(γs)
                others = reduce(vcat, (pts[j] for j in eachindex(pts) if j != i))
                cap(x) = gap_factor * sqrt(minimum(y -> sum(abs2, x - y), others))
                if any(panel_length(P) > cap(curve_eval(P, 0.0)[1]) for P in first(out[i]))
                    out[i] = discretize_curve(γs[i], p; return_params = true, lencap = cap, kw...)
                    changed = true
                end
            end
            changed || break
        end
    end
    return_params ? out : first.(out)
end
