# Automatic area mesh of Ω from its boundary panels.
#
#   build_mesh(panels, p; ...) -> (bd::BoundaryDisc, ad::AreaDisc, info)
#   mesh_quality(ad)           -> NamedTuple of quality statistics
#
# Layout of the area mesh:
#   1. One layer of "edge quads", one per boundary panel and in the same order (edge quad k
#      sits on boundary panel k, weighted = true, bpanel = k; graded layers above it, see `edge_layers`). Bottom B = the boundary panel
#      curve (t = -1 exactly on Γ), top T = straight segment between the inner vertices
#      w_k = v_k − h_k n_k and w_{k+1}. The t-direction is transverse to Γ.
#   2. The inner polygon W = (w_1, …, w_N) is meshed with straight-sided quads (weighted =
#      false, bpanel = 0) with a size function graded from the strip to the interior, by gmsh
#      (Delaunay + Blossom recombination). Remaining triangles are split into 3 quads
#      (centroid–edge midpoints).
#
# Validity of the strip: if every edge quad has J > 0 and the polygon W is simple, positively
# oriented and disjoint from Γ, then the edge quads tile the annulus between Γ and W without
# overlap (degree argument). Both conditions are checked; violations are repaired by shrinking
# the strip width h locally.

import Gmsh: gmsh

# ---------------------------------------------------------------------------------------
# Geometric predicates
# ---------------------------------------------------------------------------------------

_orient(a, b, c) = (b[1] - a[1]) * (c[2] - a[2]) - (b[2] - a[2]) * (c[1] - a[1])

function _on_segment(a, b, c)   # c collinear with a,b: is it within the bounding box?
    min(a[1], b[1]) <= c[1] <= max(a[1], b[1]) && min(a[2], b[2]) <= c[2] <= max(a[2], b[2])
end

"""Do the closed segments [a,b] and [c,d] intersect (including touching)?"""
function _segments_intersect(a, b, c, d)
    d1 = _orient(c, d, a); d2 = _orient(c, d, b)
    d3 = _orient(a, b, c); d4 = _orient(a, b, d)
    if ((d1 > 0 && d2 < 0) || (d1 < 0 && d2 > 0)) && ((d3 > 0 && d4 < 0) || (d3 < 0 && d4 > 0))
        return true
    end
    (d1 == 0 && _on_segment(c, d, a)) && return true
    (d2 == 0 && _on_segment(c, d, b)) && return true
    (d3 == 0 && _on_segment(a, b, c)) && return true
    (d4 == 0 && _on_segment(a, b, d)) && return true
    false
end

# ---------------------------------------------------------------------------------------
# Edge-quad checks
# ---------------------------------------------------------------------------------------

"""
(min J, max J) of the edge quad with bottom curve `c` and top segment wa→wb, sampled at the
points `ss`. J is affine in t, so t = ±1 suffices:
  J(s,−1) = ½ B'(s) × (T(s) − B(s)),   J(s,+1) = ½ T'(s) × (T(s) − B(s)).
"""
function _edgequad_jac_range(c::AbstractVector{Vec2}, wa::Vec2, wb::Vec2, ss)
    dT = (wb - wa) / 2
    jmin = Inf; jmax = -Inf
    for s in ss
        b, db = legendre_eval(c, s)
        T = (wa + wb) / 2 + s * dT
        d = (T - b) / 2
        j1 = _cross(db, d); j2 = _cross(dT, d)
        jmin = min(jmin, j1, j2); jmax = max(jmax, j1, j2)
    end
    jmin, jmax
end

_point_seg_dist(x, a, b) = (d = b - a; t = clamp(dot(x - a, d) / max(dot(d, d), 1e-300), 0, 1); norm(x - (a + t * d)))
_seg_dist(a, b, c, d) = _segments_intersect(a, b, c, d) ? 0.0 :
    min(_point_seg_dist(a, c, d), _point_seg_dist(b, c, d), _point_seg_dist(c, a, b), _point_seg_dist(d, a, b))

"""
Mark (in `bad`, indexed by inner vertex) the endpoints of inner-polygon segments k (from w_k
to w_{k+1}) that
  * intersect, or come closer than clearance × h̄ to, a non-adjacent inner segment, or
  * intersect the boundary polyline `G` (m segments per panel, closed, not repeated), or come
    closer than clearance × h̄ to it outside the boundary panels k−1, k, k+1,
with h̄ the mean strip width at the segment's ends. Returns the number of conflicts found.
"""
function _mark_conflicts!(bad::AbstractVector{Bool}, W::AbstractVector{Vec2}, h::AbstractVector,
                          G::AbstractVector{Vec2}, m::Int, clearance::Real,
                          nxt::AbstractVector{Int} = [mod1(k + 1, length(W)) for k in eachindex(W)],
                          prv::AbstractVector{Int} = [mod1(k - 1, length(W)) for k in eachindex(W)])
    N = length(W); M = length(G)
    # polyline successor: m points per boundary panel, closed along each curve
    gnext(j) = (q = (j - 1) ÷ m + 1; jj = (j - 1) % m; jj < m - 1 ? j + 1 : (nxt[q] - 1) * m + 1)
    seglen = [norm(W[nxt[k]] - W[k]) for k in 1:N]
    hcell = max(sort(seglen)[cld(N, 2)], 1e-300)
    grid = _BucketGrid(hcell, Dict{Tuple{Int,Int},Vector{Int}}())
    for k in 1:N
        a = W[k]; b = W[nxt[k]]
        _insert_box!(grid, k, min.(a, b), max.(a, b))
    end
    for j in 1:M
        a = G[j]; b = G[gnext(j)]
        _insert_box!(grid, -j, min.(a, b), max.(a, b))
    end
    r = [clearance * (h[k] + h[nxt[k]]) / 2 for k in 1:N]
    nbad = 0
    for k in 1:N
        k1 = nxt[k]
        a = W[k]; b = W[k1]
        _foreach_in_box(grid, min.(a, b) .- r[k], max.(a, b) .+ r[k]) do id
            if id > 0
                id > k || return                       # each W–W pair once
                (id == k1 || nxt[id] == k) && return     # adjacent
                if _seg_dist(a, b, W[id], W[nxt[id]]) < (r[k] + r[id]) / 2
                    bad[k] = bad[k1] = bad[id] = bad[nxt[id]] = true
                    nbad += 1
                end
            else
                j = -id
                q = (j - 1) ÷ m + 1                    # boundary panel of this polyline piece
                near = q == k || q == k1 || q == prv[k]
                c = G[j]; d = G[gnext(j)]
                if near ? _segments_intersect(a, b, c, d) : _seg_dist(a, b, c, d) < r[k]
                    bad[k] = bad[k1] = true
                    nbad += 1
                end
            end
        end
    end
    nbad
end

# ---------------------------------------------------------------------------------------
# Mesh construction
# ---------------------------------------------------------------------------------------

"""
    build_mesh(panels, p; strip_factor=0.5, thick_factor=0.35, interior_size=Inf,
               size_factor=1.5, grading=0.5, jtol=0.05, shrink=0.8, clearance=0.5, maxfix=60,
               gmsh_options=(), edge_layers=3, edge_grading=0.2, coarse_interior=false)  ->  (bd, ad, info)

Build the boundary discretization `bd = BoundaryDisc(panels, p)` and the area discretization
`ad::AreaDisc` of the domain enclosed by the counterclockwise panels.

Edge quads: panels 1:N of `ad` (N = length(panels)), edge quad k on boundary panel k. The strip
width at boundary vertex v_k (start of panel k) is

    h_k = min( strip_factor × min(L_{k−1}, L_k),  thick_factor × lfs(v_k) )

(L = panel lengths, lfs = interior local feature size from `local_feature_size`; thick_factor ≈ 0.35 keeps strips from
opposite sides of a neck of half-width δ ≈ lfs apart), but at least 1.5× the inward bulge of
the adjacent panels beyond their chords (so the top segment clears a concave bottom curve).
h is then shrunk locally (×`shrink`, not below that bound) until every edge quad has
min J > jtol × max J (checked at t = ±1 on a 3p-point Chebyshev grid in s; J is affine in t),
top segments are not collapsed (≥ 0.2 × chord), and the inner polygon W is simple, does not
cross Γ, and keeps a distance ≥ clearance × h from non-neighboring parts of W and Γ (so the
interior region never gets pinched). If that fails, an error asks for boundary refinement.

Interior: gmsh Delaunay triangulation of W with target edge length

    f(x) = min( interior_size,  min_k ( size_factor × ℓ_k + grading × |x − w_k| ) ),

ℓ_k the mean W-edge length at w_k, recombined into quads by Blossom matching (W vertices kept as
boundary nodes; for odd N one W edge gets a midpoint). Leftover triangles and non-convex quads
(rare) are split into 3 quads each (centroid–edge midpoints). `gmsh_options` (name => value
pairs) override gmsh options.

Coarse interior (`coarse_interior = true`): every other inner vertex w_k (k even within each
curve of ≥ 6 panels) is placed at the midpoint of w_{k−1} and w_{k+1}, so edge quads k−1 and k
share one straight top segment, and gmsh meshes the polygon without these vertices (half as many
boundary edges, so ℓ_k and the interior panels about double next to the strip; the interior is
non-conforming at the dropped vertices, still an exact tiling). The strip checks above apply
to the moved vertices; if the repair fails that way, or the panels' total area differs from the
curves' (a tiling check), the full polygon is used. A dumbbell-shaped domain at tol 1e-6: 83.5k → 58.9k area
nodes, same GMRES iterations and discretization error; the saving depends on gmsh's
recombination (none for some meshes), hence off by default.

Edge layers: with `edge_layers = L > 1` every edge quad is split in t into L layers with
interfaces at (1+t)/2 = σ^{L−1}, …, σ (σ = `edge_grading`), i.e. geometrically graded toward Γ.
Only the layer on Γ is weighted (panels 1..N); layer j ≥ 2 of edge quad k is panel N + (j−2)N + k
(unweighted). All layers of edge quad k have bpanel = k (`panel_groups` groups them, and `local_blocks` gives
an evaluator's dense block on each group, for a block-Jacobi preconditioner). The graded layers resolve the d^{1/2} log d edge terms (tangential forcing); default 3
layers at σ = 0.2.

`info` fields: h, W (inner vertices), lfs (interior lfs at boundary vertices), L (panel lengths), nedge,
edge_layers, edge_grading,
ninterior, nmerged (interior quads), nsplit (triangles split into 3 quads), ntriangles, tri_points,
triangles, nsteiner_W (extra nodes on W), nfix (repair iterations), ndropped (inner vertices left out of
the interior by `coarse_interior`), times (NamedTuple).
"""
function build_mesh(panels::Vector{CurvePanel}, p::Int; holes::AbstractVector = Vector{CurvePanel}[],
                    strip_factor::Real = 0.5, thick_factor::Real = 0.35,
                    interior_size::Real = Inf, size_factor::Real = 1.5, grading::Real = 0.5,
                    jtol::Real = 0.05, shrink::Real = 0.8, clearance::Real = 0.5, maxfix::Int = 60, gmsh_options = (),
                    edge_layers::Integer = 3, edge_grading::Real = 0.2, coarse_interior::Bool = false)
    edge_layers >= 1 || error("build_mesh: edge_layers must be ≥ 1")
    0 < edge_grading < 1 || error("build_mesh: edge_grading must be in (0, 1)")
    t_start = time_ns()
    curves = Vector{CurvePanel}[panels]
    append!(curves, [collect(CurvePanel, h) for h in holes])
    all(c -> length(c) >= 3, curves) || error("build_mesh: need at least 3 panels per curve")
    _check_holes(curves)
    cstart = cumsum([1; length.(curves)])
    panels = reduce(vcat, curves)
    N = length(panels)
    nxt, prv = cyclic_maps(cstart)
    bd = BoundaryDisc(panels, p; cstart = cstart)
    L = [panel_length(P) for P in panels]

    # boundary vertices and (averaged) normals
    v = Vector{Vec2}(undef, N); nrm = Vector{Vec2}(undef, N)
    for k in 1:N
        g, dg = curve_eval(panels[k], -1.0)
        _, dgp = curve_eval(panels[prv[k]], 1.0)
        τ = normalize(normalize(dg) + normalize(dgp))
        v[k] = g; nrm[k] = Vec2(τ[2], -τ[1])
    end

    # strip widths (feature size over all curves, so the width of an annulus is seen)
    lfs = local_feature_size(panels; radius_factor = 3 * strip_factor / thick_factor, cstart = cstart).vertex_interior
    h = [min(strip_factor * min(L[k], L[prv[k]]), thick_factor * lfs[k]) for k in 1:N]
    t_lfs = time_ns()

    # repair loop: positive Jacobians, simple inner polygons not crossing Γ or each other
    ns = 3p
    ss = [cos(π * (i - 1) / (ns - 1)) for i in 1:ns]
    mG = max(p, 8)
    G = curve_polyline(panels, mG)
    # Lower bound for h: a concave panel bulges inward past its chord by `bulge`; the top
    # segment must stay clear of it, so h ≥ 1.5 × bulge at both of its vertices.
    bulge = zeros(N)
    for k in 1:N
        a = v[k]; d = v[nxt[k]] - a; d = d / norm(d)
        for s in ss
            bulge[k] = max(bulge[k], _cross(d, curve_eval(panels[k], s)[1] - a))
        end
    end
    hlo = [1.5 * max(bulge[k], bulge[prv[k]]) for k in 1:N]
    h .= max.(h, hlo)
    W = Vector{Vec2}(undef, N)
    bad = falses(N)
    nfix = 0
    # coarse interior: inner vertices at the midpoint of their neighbors (dropped from the interior)
    mid = falses(N)
    if coarse_interior
        for c in 1:length(cstart)-1
            M = cstart[c+1] - cstart[c]
            M >= 6 || continue
            for j in 2:2:(iseven(M) ? M : M - 1)
                mid[cstart[c]+j-1] = true
            end
        end
    end
    h0 = copy(h)
    iter = 0
    while true
        iter += 1
        @. W = v - h * nrm
        for k in 1:N
            mid[k] && (W[k] = (W[prv[k]] + W[nxt[k]]) / 2)
        end
        fill!(bad, false)
        for k in 1:N
            k1 = nxt[k]
            jmin, jmax = _edgequad_jac_range(panels[k].c, W[k], W[k1], ss)
            # positive Jacobian (with margin), and no collapsed top segment
            if !(jmin > jtol * jmax) || norm(W[k1] - W[k]) < 0.2 * norm(v[k1] - v[k])
                bad[k] = true; bad[k1] = true
            end
        end
        _mark_conflicts!(bad, W, h, G, mG, clearance, nxt, prv)
        any(bad) || break
        # a dropped vertex follows its neighbors: mark them instead
        for k in 1:N
            mid[k] && bad[k] && (bad[prv[k]] = true; bad[nxt[k]] = true)
        end
        stuck = all(h[k] <= hlo[k] for k in 1:N if bad[k] && !mid[k])
        if iter == maxfix || stuck
            if any(mid)                      # retry with the full polygon
                fill!(mid, false); h .= h0; iter = 0; nfix = 0
                continue
            end
            error("build_mesh: could not produce a valid strip after $nfix repairs " *
                  "($(count(bad)) vertices still bad, e.g. boundary vertex $(findfirst(bad)) at " *
                  "$(v[findfirst(bad)])); refine the boundary panels there (smaller lfs_factor or maxlen in discretize_curve)")
        end
        for k in 1:N
            bad[k] && (h[k] = max(shrink * h[k], hlo[k]))
        end
        nfix += 1
    end
    for c in eachindex(curves)
        A = polygon_area(W[cstart[c]:cstart[c+1]-1])
        (c == 1 ? A > 0 : A < 0) || error("build_mesh: inner polygon of curve $c has the wrong orientation")
    end
    t_strip = time_ns()

    # interior of the region bounded by the inner polygons (outer loop first, then holes), without
    # the dropped vertices
    keep = findall(.!mid)
    Wi = W[keep]
    cstart_i = [1; [count(.!mid[1:cstart[c+1]-1]) + 1 for c in 1:length(cstart)-1]]
    nxt_i, prv_i = cyclic_maps(cstart_i)
    ℓ = [0.5 * (norm(Wi[k] - Wi[prv_i[k]]) + norm(Wi[nxt_i[k]] - Wi[k])) for k in eachindex(Wi)]
    it = _interior_gmsh(Wi, ℓ; loops = cstart_i, interior_size, size_factor, grading, options = gmsh_options)
    tpts, quads, tris = it.pts, it.quads, it.tris
    t_tri = time_ns()

    # panels: edge quads (weighted layer on Γ, then the other edge layers), interior quads,
    # and 3 quads per remaining triangle
    nl = edge_layers
    # layer interfaces (1+t)/2 = σ^{L−1}, …, σ, 1 of the full edge quad (geometric grading)
    ζ = [Float64(edge_grading)^(nl - j) for j in 1:nl]
    NE = N * nl
    aps = Vector{Panel}(undef, NE + length(quads) + 3length(tris))
    for k in 1:N
        Bc = panels[k].c; Tc = segment_coefs(W[k], W[nxt[k]])
        # x(s, (1+t)/2 = z) = (1 − z) B(s) + z T(s), as Legendre coefficients
        lev(z) = [(1 - z) * (i <= length(Bc) ? Bc[i] : zero(Vec2)) + z * (i <= length(Tc) ? Tc[i] : zero(Vec2))
                  for i in 1:max(length(Bc), length(Tc))]
        aps[k] = Panel(copy(Bc), nl == 1 ? Tc : lev(ζ[1]), true, k)
        for j in 2:nl
            aps[N+(j-2)*N+k] = Panel(lev(ζ[j-1]), j == nl ? Tc : lev(ζ[j]), false, k)
        end
    end
    # quad with counterclockwise corners c1..c4: B = c1→c2 (t = −1), T = c4→c3 (t = +1)
    for (m, q) in enumerate(quads)
        c1, c2, c3, c4 = tpts[q[1]], tpts[q[2]], tpts[q[3]], tpts[q[4]]
        aps[NE+m] = Panel(segment_coefs(c1, c2), segment_coefs(c4, c3), false, 0)
    end
    for (m, (i, j, k)) in enumerate(tris)
        a, b, c = tpts[i], tpts[j], tpts[k]
        g = (a + b + c) / 3
        mab = (a + b) / 2; mbc = (b + c) / 2; mca = (c + a) / 2
        base = NE + length(quads) + 3(m - 1)
        # quad at vertex x: corners x, m(x,y), g, m(z,x) (counterclockwise)
        aps[base+1] = Panel(segment_coefs(a, mab), segment_coefs(mca, g), false, 0)
        aps[base+2] = Panel(segment_coefs(b, mbc), segment_coefs(mab, g), false, 0)
        aps[base+3] = Panel(segment_coefs(c, mca), segment_coefs(mbc, g), false, 0)
    end
    # coarse interior: check the tiling and fall back to the full polygon if it fails (seen once: an
    # eccentric hole with 16 panels per curve)
    if any(mid) && !_tiles(aps, panels, W, nxt, p)
        return build_mesh(curves[1], p; holes = curves[2:end], strip_factor, thick_factor, interior_size,
                          size_factor, grading, jtol, shrink, clearance, maxfix, gmsh_options, edge_layers,
                          edge_grading, coarse_interior = false)
    end
    ad = AreaDisc(aps, p)
    t_end = time_ns()

    times = (lfs = (t_lfs - t_start) / 1e9, strip = (t_strip - t_lfs) / 1e9,
             triangulate = (t_tri - t_strip) / 1e9, assemble = (t_end - t_tri) / 1e9,
             total = (t_end - t_start) / 1e9)
    info = (h = h, W = W, lfs = lfs, L = L, nedge = N, cstart = cstart, edge_layers = edge_layers, edge_grading = edge_grading,
            ninterior = length(quads) + 3length(tris), nmerged = length(quads), nsplit = length(tris),
            ntriangles = it.ntriangles, tri_points = tpts, triangles = it.triangles,
            nsteiner_W = it.nsteiner_W, nfix = nfix, ndropped = count(mid), times = times)
    return bd, ad, info
end

"""
Do the area panels `aps` tile the region between the boundary panels and the inner polygon W
exactly? Their area (p-point Gauss–Legendre, exact for these Jacobians) must equal the area
enclosed by the panel curves, corrected for the gaps at panel junctions (consecutive panels
need not meet exactly; the edge quads leave out the triangle end_k, start_{k+1}, w_{k+1}):
Σ curve areas + ½ Σ_k (end_k − start_{k+1}) × w_{k+1}.
"""
function _tiles(aps, panels, W, nxt, p; rtol = 1e-10)
    r = gl_rule(p)
    Am = sum(r.w[i] * r.w[j] * panel_map(P, r.x[i], r.x[j])[4] for P in aps, i in 1:p, j in 1:p)
    Aex = curve_area(panels) +
          sum(_cross(curve_eval(panels[k], 1.0)[1] - curve_eval(panels[nxt[k]], -1.0)[1], W[nxt[k]]) / 2 for k in eachindex(panels))
    abs(Am - Aex) <= rtol * abs(Aex)
end

# one drop: the outer curve (counterclockwise, positive area) and holes (clockwise) that lie
# inside it and are disjoint from each other
function _check_holes(curves; m::Int = 16)
    curve_area(curves[1]) > 0 || error("build_mesh: the outer curve must be counterclockwise")
    G = [curve_polyline(c, m) for c in curves]
    for c in 2:length(curves)
        curve_area(curves[c]) < 0 || error("build_mesh: hole $(c - 1) must be clockwise (see reverse_curve)")
        all(x -> point_in_polygon(x, G[1]), G[c]) || error("build_mesh: hole $(c - 1) is not inside the outer curve")
        for d in 2:c-1
            (any(x -> point_in_polygon(x, G[d]), G[c]) || any(x -> point_in_polygon(x, G[c]), G[d])) &&
                error("build_mesh: holes $(d - 1) and $(c - 1) overlap or are nested")
        end
    end
end

"""
    build_mesh(curves::Vector{Vector{CurvePanel}}, p; kw...) -> (bd, ad, infos)
    build_mesh(drops::Vector{Vector{Vector{CurvePanel}}}, p; kw...) -> (bd, ad, infos)

Several disjoint drops. In the first form each drop is one closed counterclockwise curve; in
the second, drop i is `drops[i] = [outer, hole₁, …]` (outer counterclockwise, holes clockwise).
Each drop is meshed by the single-drop `build_mesh` (same keywords) and the results are
concatenated: drop i owns the boundary panels `drop_panels(bd, i)` (its curves
`drop_curves(bd, i)`, outer first) and the area panels `drop_panels(ad, i)`, and the edge
columns' `bpanel` refer to the global boundary panel numbering. `infos[i]` is drop i's info.

The drops are meshed independently: boundary panels facing a narrow gap are not refined for
the other drop here (choose them so with `discretize_curves`).
"""
build_mesh(curves::AbstractVector{<:AbstractVector{CurvePanel}}, p::Int; kw...) =
    build_mesh([[collect(CurvePanel, c)] for c in curves], p; kw...)

"""
    film_drops(panels; holes=[]) -> Vector{Vector{Vector{CurvePanel}}}

The canonical description of a film: a list of drops, each `[outer, hole₁, …]` (outer
counterclockwise, holes clockwise). Accepts one curve (with `holes`), a list of curves (one drop
each), or the canonical form itself.
"""
film_drops(c::AbstractVector{CurvePanel}; holes::AbstractVector=Vector{CurvePanel}[]) =
    [Vector{CurvePanel}[collect(CurvePanel, c), (collect(CurvePanel, h) for h in holes)...]]
function film_drops(cs::AbstractVector{<:AbstractVector{CurvePanel}}; holes::AbstractVector=Vector{CurvePanel}[])
    isempty(holes) || error("film_drops: `holes` goes with a single outer curve; give drops as [[outer, holes…], …]")
    [Vector{CurvePanel}[collect(CurvePanel, c)] for c in cs]
end
function film_drops(ds::AbstractVector{<:AbstractVector{<:AbstractVector{CurvePanel}}}; holes::AbstractVector=Vector{CurvePanel}[])
    isempty(holes) || error("film_drops: `holes` goes with a single outer curve; give drops as [[outer, holes…], …]")
    [Vector{CurvePanel}[collect(CurvePanel, c) for c in d] for d in ds]
end

function build_mesh(drops::AbstractVector{<:AbstractVector{<:AbstractVector{CurvePanel}}}, p::Int; kw...)
    isempty(drops) && error("build_mesh: no drops")
    _check_disjoint([d[1] for d in drops])
    parts = [build_mesh(collect(CurvePanel, d[1]), p; holes = [collect(CurvePanel, h) for h in d[2:end]], kw...) for d in drops]
    bpan = CurvePanel[]; apan = Panel[]; bstart = [1]; cstart = [1]; astart = [1]
    for (bd, ad, _) in parts
        off = length(bpan)
        append!(bpan, bd.panels)
        append!(cstart, bd.cstart[2:end] .+ off)
        append!(apan, [Panel(P.B, P.T, P.weighted, P.bpanel == 0 ? 0 : P.bpanel + off) for P in ad.panels])
        push!(bstart, length(bpan) + 1); push!(astart, length(apan) + 1)
    end
    BoundaryDisc(bpan, p; pstart = bstart, cstart = cstart), AreaDisc(apan, p; pstart = astart), [part[3] for part in parts]
end

# drops must not overlap or nest: no point of one outer curve's polyline inside another's, and no crossings
function _check_disjoint(curves; m::Int = 16)
    G = [curve_polyline(c, m) for c in curves]
    for i in eachindex(G), j in eachindex(G)
        i == j && continue
        any(x -> point_in_polygon(x, G[i]), G[j]) && error("build_mesh: curves $i and $j overlap or are nested")
    end
end

# gmsh's mesh-size callback must be a plain function (closures are not supported by
# @cfunction on all platforms), so its data lives here; calls are serialized by _GMSH_LOCK.
const _GMSH_LOCK = ReentrantLock()
const _GMSH_W = Ref(Vec2[])
const _GMSH_L = Ref(Float64[])
const _GMSH_PAR = Ref((Inf, 1.5, 0.5))     # (interior_size, size_factor, grading)

function _gmsh_size(dim, tag, x, y, z, lc)
    W = _GMSH_W[]; ℓ = _GMSH_L[]; f0, sf, gr = _GMSH_PAR[]
    f = f0
    @inbounds for k in eachindex(W)
        f = min(f, sf * ℓ[k] + gr * hypot(W[k][1] - x, W[k][2] - y))
    end
    f
end

"""
Interior by gmsh: Delaunay triangulation with the size function f(x) of `build_mesh`,
recombined into quads by the Blossom perfect-matching algorithm. The vertices of W are kept
as the only boundary nodes (one mesh edge per W edge), except that for odd N the longest W
edge gets its midpoint (a quad mesh of a polygon needs an even number of boundary edges); the
edge quads are unaffected (non-conforming, still an exact tiling). Returned quads are
counterclockwise and convex; non-convex quads (rare) are returned as two triangles, and
triangles left by the recombination are returned in `tris` (split into 3 quads by the caller).
Recombination is limited to quads of gmsh quality ≥ 0.4 (worst interior aspect ratio ≈ 4 on
the test domains, vs > 10 without the limit). `options` (name => value pairs) override any
gmsh option.
"""
function _interior_gmsh(W::Vector{Vec2}, ℓ::Vector{Float64}; loops::AbstractVector{Int} = [1, length(W) + 1],
                        interior_size, size_factor, grading, options = ())
    N = length(W)
    polys = [W[loops[c]:loops[c+1]-1] for c in 1:length(loops)-1]   # outer loop first, then holes
    if isodd(N)    # a quad mesh needs an even number of boundary edges in total
        lens = [(c, k, norm(P[mod1(k + 1, length(P))] - P[k])) for (c, P) in enumerate(polys) for k in eachindex(P)]
        c, k, _ = lens[argmax(last.(lens))]
        P = polys[c]; insert!(P, k + 1, (P[k] + P[mod1(k + 1, length(P))]) / 2)
    end
    pts = Vec2[]; quads = NTuple{4,Int}[]; tris = NTuple{3,Int}[]
    lock(_GMSH_LOCK) do
        _GMSH_W[] = W; _GMSH_L[] = ℓ; _GMSH_PAR[] = (Float64(interior_size), Float64(size_factor), Float64(grading))
        gmsh.initialize(String[], false, false)
        try
            gmsh.option.setNumber("General.Terminal", 0)
            gmsh.option.setNumber("General.Verbosity", 1)
            gmsh.model.add("interior")
            cl = Int32[]; lines = Int32[]
            for P in polys
                M = length(P)
                pt = [gmsh.model.geo.addPoint(x[1], x[2], 0.0) for x in P]
                ln = [gmsh.model.geo.addLine(pt[k], pt[mod1(k + 1, M)]) for k in 1:M]
                push!(cl, gmsh.model.geo.addCurveLoop(ln)); append!(lines, ln)
            end
            gmsh.model.geo.addPlaneSurface(cl)
            gmsh.model.geo.synchronize()
            for l in lines
                gmsh.model.mesh.setTransfiniteCurve(l, 2)
            end
            gmsh.model.mesh.setSizeCallback(_gmsh_size)
            gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
            gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
            gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
            gmsh.option.setNumber("Mesh.Algorithm", 5)                 # Delaunay
            gmsh.option.setNumber("Mesh.RecombineAll", 1)
            gmsh.option.setNumber("Mesh.RecombinationAlgorithm", 1)    # Blossom
            # pairs whose quad would have quality < 0.4 stay triangles (split into 3 quads by
            # the caller); without this Blossom accepts kites with aspect ratio > 10
            gmsh.option.setNumber("Mesh.RecombineMinimumQuality", 0.4)
            for (k, v) in options
                gmsh.option.setNumber(String(k), Float64(v))
            end
            gmsh.model.mesh.generate(2)
            tags, coord, _ = gmsh.model.mesh.getNodes()
            idx = Dict(Int(t) => i for (i, t) in enumerate(tags))
            for i in eachindex(tags)
                push!(pts, Vec2(coord[3i-2], coord[3i-1]))
            end
            types, _, enodes = gmsh.model.mesh.getElements(2)
            for (ty, en) in zip(types, enodes)
                if ty == 3
                    for j in 1:4:length(en)
                        push!(quads, ntuple(i -> idx[Int(en[j+i-1])], 4))
                    end
                elseif ty == 2
                    for j in 1:3:length(en)
                        push!(tris, ntuple(i -> idx[Int(en[j+i-1])], 3))
                    end
                else
                    error("unexpected gmsh element type $ty")
                end
            end
        finally
            gmsh.finalize()
        end
    end
    # orientation and convexity
    good = NTuple{4,Int}[]
    for q in quads
        c = [pts[i] for i in q]
        polygon_area(c) < 0 && (q = (q[1], q[4], q[3], q[2]); c = [pts[i] for i in q])
        cr = [_cross(c[mod1(i + 1, 4)] - c[i], c[mod1(i + 2, 4)] - c[mod1(i + 1, 4)]) for i in 1:4]
        if all(>(0), cr)
            push!(good, q)
        else                                   # split through the reflex vertex
            r = findfirst(<=(0), cr) % 4 + 1   # cr[i] is the turn at vertex i+1
            q2 = ntuple(i -> q[mod1(r + i - 1, 4)], 4)
            push!(tris, (q2[1], q2[2], q2[3])); push!(tris, (q2[1], q2[3], q2[4]))
        end
    end
    tris = [_orient(pts[t[1]], pts[t[2]], pts[t[3]]) > 0 ? t : (t[1], t[3], t[2]) for t in tris]
    (pts = pts, quads = good, tris = tris, ntriangles = length(tris), triangles = tris,
     nsteiner_W = sum(length, polys) - N)
end

# ---------------------------------------------------------------------------------------
# Quality report
# ---------------------------------------------------------------------------------------

"""
    mesh_quality(ad; ng=2p)

Per-panel Jacobian ratio min J / max J and aspect ratio max σ₁/σ₂ of [∂ₛx ∂ₜx] (both sampled
on an ng×ng Chebyshev grid including the panel edges), and panel diameters. Returns a
NamedTuple with summary statistics (`jratio_min`, `aspect_max`, ...), counts, and the
per-panel arrays `jratio`, `aspect`, `diam`.
"""
function mesh_quality(ad::AreaDisc; ng::Int = 2ad.p)
    P = ad.panels; n = length(P)
    g = [cos(π * (i - 1) / (ng - 1)) for i in 1:ng]
    jr = zeros(n); asp = zeros(n); diam = zeros(n)
    Jmin_all = Inf
    for (k, Q) in enumerate(P)
        jmin = Inf; jmax = -Inf; a = 1.0
        for s in g, t in g
            _, xs, xt, J = panel_map(Q, s, t)
            jmin = min(jmin, J); jmax = max(jmax, J)
            σ = svdvals(hcat(xs, xt))
            a = max(a, σ[1] / σ[2])
        end
        jr[k] = jmin / jmax; asp[k] = a; diam[k] = panel_diameter(Q)
        Jmin_all = min(Jmin_all, jmin)
    end
    isedge = [Q.bpanel > 0 for Q in P]                      # edge columns (all layers)
    ne = count(isedge)
    med(x) = isempty(x) ? NaN : sort(x)[cld(length(x), 2)]
    (npanels = n, nedge = ne, ninterior = n - ne, nnodes = length(ad.x),
     J_min = Jmin_all, jratio_min = minimum(jr), jratio_median = med(jr),
     aspect_max = maximum(asp), aspect_median = med(asp),
     aspect_max_edge = ne > 0 ? maximum(asp[isedge]) : NaN,
     aspect_max_interior = ne < n ? maximum(asp[.!isedge]) : NaN,
     diam_min = minimum(diam), diam_max = maximum(diam),
     jratio = jr, aspect = asp, diam = diam)
end
