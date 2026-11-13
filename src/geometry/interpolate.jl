# Interpolation of nodal values on an area mesh at arbitrary points (locating each point in a panel by its Newton
# preimage, `panel_preimage` in src/quadrature/area_quad.jl).

"""
    interpolate_nodal(ad, v, pts; outside=nothing) -> values at pts

Evaluate the per-panel tensor interpolant of nodal values `v` (no edge-weight conversion) at
arbitrary points; points outside every panel get `outside`.
"""
function interpolate_nodal(ad0::AreaDisc, v0::AbstractVector{T}, pts::AbstractVector{Vec2}; outside=nothing) where {T}
    np0 = npanels(ad0); p = ad0.p; p2 = p * p
    lo = Vector{Vec2}(undef, np0); hi = Vector{Vec2}(undef, np0)
    for (k, P) in enumerate(ad0.panels)
        q = [panel_map(P, s, t)[1] for s in range(-1, 1, length=7) for t in (-1.0, 1.0)]
        append!(q, [panel_map(P, s, t)[1] for t in range(-1, 1, length=7) for s in (-1.0, 1.0)])
        a = reduce((u, w) -> min.(u, w), q); b = reduce((u, w) -> max.(u, w), q)
        pad = 0.05 * norm(b - a); lo[k] = a .- pad; hi[k] = b .+ pad
    end
    glo = reduce((u, w) -> min.(u, w), lo); ghi = reduce((u, w) -> max.(u, w), hi)
    hcell = sqrt(prod(ghi - glo) / np0)
    cell(x) = (floor(Int, (x[1] - glo[1]) / hcell), floor(Int, (x[2] - glo[2]) / hcell))
    bins = Dict{Tuple{Int,Int},Vector{Int}}()
    for k in 1:np0
        (i0, j0) = cell(lo[k]); (i1, j1) = cell(hi[k])
        for i in i0:i1, j in j0:j1
            push!(get!(bins, (i, j), Int[]), k)
        end
    end
    out = Vector{Union{T,typeof(outside)}}(undef, length(pts))
    Threads.@threads for i in eachindex(pts)
        x = pts[i]; val = outside
        for k in get(bins, cell(x), Int[])
            (all(x .>= lo[k]) && all(x .<= hi[k])) || continue
            s, t, res = panel_preimage(ad0.panels[k], x)
            (abs(s) <= 1 + 1e-9 && abs(t) <= 1 + 1e-9 && res <= 1e-9 * (1 + norm(x))) || continue
            ls = lagrange_row(ad0.rs, clamp(s, -1.0, 1.0)); lt = lagrange_row(trule(ad0, k), clamp(t, -1.0, 1.0))
            acc = zero(T); off = (k - 1) * p2
            for jj in 1:p, ii in 1:p
                acc += (ls[ii] * lt[jj]) * v0[off+ii+(jj-1)*p]
            end
            val = acc
            break
        end
        out[i] = val
    end
    out
end
