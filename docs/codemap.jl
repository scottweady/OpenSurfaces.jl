# OpenSurfaces' code map: which functions of src/ each feature runs. Every setting (a kernel family evaluated by
# `Evaluator`, or an option: sources and targets, fast sums, operators, geometry, boundary quadrature, building blocks)
# is one small run, changed from the base case (`Evaluator(Stokeslet3D(), ad)` on the unit disc at p = 8, one apply)
# in that setting only, run under line coverage; the lines it executes are mapped back to the top-level function
# definitions of src/. Written as data for src/assets/codemap/codemap.js, which draws the docs page "Code map".
#   julia --project=docs docs/codemap.jl [covdir]          (from the package root; a few minutes)
# The script runs itself again in a child Julia with coverage on (no package image, so the child compiles
# OpenSurfaces), on one thread: the threaded loops run the same lines, and with several threads updating the same
# coverage counters the runs were eight times slower (Evaluator(Stokeslet3D(), ad): 44 s on 8 threads, 5.4 s on 1).
# The child's tracefiles (<setting>.info) go to `covdir` if given, and a `covdir` that already holds them skips the
# run.
const ROOT = dirname(@__DIR__)                       # the package root
const SRC = joinpath(ROOT, "src")
const OUT = joinpath(@__DIR__, "src", "assets", "codemap", "codemap_data.js")

# ---- the data every setting starts from: built once, before the coverage of the first setting is cleared ---------
# The base mesh (the unit disc at p = 8: 16 boundary panels, 128 boundary nodes, 88 area panels, 5632 area nodes),
# densities on it, points inside, and a smaller mesh (an ellipse at p = 6, one edge layer) for the dense matrices.
# The base case builds the disc's mesh again inside its own run, so the map shows what that costs; the other settings
# use this one.
const SETUP = quote
    using OpenSurfaces, LinearAlgebra
    disc = θ -> Vec2(cos(θ), sin(θ))
    pans = discretize_curve(disc, 8)
    bd, ad, _ = build_mesh(pans, 8)
    F = [Vec2(x[2], 1 - x[1]^2) for x in ad.x]           # a Vec2 density at the area nodes
    g = [1 + x[1] * x[2] for x in ad.x]                   # a scalar one
    σ = [x[1] for x in bd.x]                              # densities at the boundary nodes
    ν = [Vec2(-x[2], x[1]) for x in bd.x]
    pts = [0.5 * Vec2(cos(t), sin(t)) for t in range(0, 2π; length=41)[1:40]]
    bds, ads, _ = build_mesh(discretize_curve(disc, 6; tol=1e-4), 6; edge_layers=1)
    Fs = [Vec2(x[2], 1 - x[1]^2) for x in ads.x]
end

# ---- the settings: (id, label, group, what it is, the run) ----------------------------------------------------
# `parent` is the kernel an option is compared with (and run with, when it builds an Evaluator); "" for the kernels.
# `schemes`: the kernels the option applies to, each combination checked by running it.
const AREA = ["stokes3d", "laplace3d", "laplace2d", "biharmonic", "mh", "callable"]      # area sources
const ALL = [AREA; "line"]
const SETTINGS = [
    # the kernel families
    (id="stokes3d", label="Stokeslet3D", group="Kernel", base=true, parent="", schemes=String[],
     about="Evaluator(Stokeslet3D(), ad): the unit disc from discretize_curve and build_mesh at p = 8 (5632 area nodes), area sources at their own nodes, one apply to a Vec2 density, summed by Ewald; the base case, and the only setting that builds its mesh",
     run=:(let pans = discretize_curve(disc, 8), (bd, ad, _) = build_mesh(pans, 8)
               Evaluator(Stokeslet3D(), ad)([Vec2(x[2], 1 - x[1]^2) for x in ad.x]) end)),
    (id="laplace3d", label="LaplaceKernel3D", group="Kernel", base=false, parent="", schemes=String[],
     about="Evaluator(LaplaceKernel3D(), ad), the kernel 1/(4πρ), on a scalar density: summed by FMM3D's Laplace FMM",
     run=:(Evaluator(LaplaceKernel3D(), ad)(g))),
    (id="laplace2d", label="LaplaceKernel2D", group="Kernel", base=false, parent="", schemes=String[],
     about="Evaluator(LaplaceKernel2D(), ad), the kernel (1/2π) log ρ, and LaplaceGradient2D, its gradient: summed by FMM2D's Laplace FMM",
     run=:(begin Evaluator(LaplaceKernel2D(), ad)(g); Evaluator(LaplaceGradient2D(), ad)(g) end)),
    (id="biharmonic", label="BiharmonicGradient2D", group="Kernel", base=false, parent="", schemes=String[],
     about="Evaluator(BiharmonicGradient2D(; κ = 0.3), ad), the gradient of the bilaplacian kernel, a vector potential: FMM2D's Laplace FMM on three charge vectors",
     run=:(Evaluator(BiharmonicGradient2D(; κ=0.3), ad)(g))),
    (id="mh", label="MHKernel", group="Kernel", base=false, parent="", schemes=String[],
     about="Evaluator(MHKernel(3.0), ad), the modified Helmholtz kernel K₀(3ρ)/2π: FMM2D's Helmholtz FMM at an imaginary wavenumber",
     run=:(Evaluator(MHKernel(3.0), ad)(g))),
    (id="callable", label="A function", group="Kernel", base=false, parent="", schemes=String[],
     about="Evaluator(r -> exp(-norm(r)) / (4π * norm(r)), ad): a plain function as the kernel, with no fast sum: the threaded direct sum, and the near corrections of the function itself",
     run=:(Evaluator(r -> exp(-norm(r)) / (4π * norm(r)), ad)(g))),
    (id="line", label="Line sources", group="Kernel", base=false, parent="", schemes=String[],
     about="Evaluator(LaplaceKernel2D(), bd; targets = ad.x): sources on the boundary nodes (a BoundaryDisc), the log potential at the area nodes; LineQuad's corrections and a direct sum",
     run=:(Evaluator(LaplaceKernel2D(), bd; targets=ad.x)(σ))),
    # sources and targets
    (id="smooth", label="Smooth density", group="Sources & targets", base=false, parent="stokes3d", schemes=AREA,
     about="density = :smooth: physical nodal values of a smooth density, integrated on the unweighted copy of the mesh",
     run=:(Evaluator(Stokeslet3D(), ad; density=:smooth)(F))),
    (id="boundary", label="Targets on Γ", group="Sources & targets", base=false, parent="stokes3d", schemes=AREA,
     about="targets = :boundary (with bd): the potential at the boundary nodes, which lie on the edges of the area panels (boundary_target_info)",
     run=:(Evaluator(Stokeslet3D(), ad; targets=:boundary, bd)(F))),
    (id="points", label="Targets at points", group="Sources & targets", base=false, parent="stokes3d", schemes=ALL,
     about="targets = a vector of 40 points inside the disc",
     run=:(Evaluator(Stokeslet3D(), ad; targets=pts)(F))),
    (id="sets", label="Several target sets", group="Sources & targets", base=false, parent="laplace3d",
     schemes=["laplace3d", "laplace2d", "biharmonic", "mh", "callable", "line"],
     about="targets = (:area, :boundary, points), here with LaplaceKernel3D: one fast sum, one output per set (Stokeslet3D's Ewald plan takes one set)",
     run=:(Evaluator(LaplaceKernel3D(), ad; targets=(:area, :boundary, pts), bd)(g))),
    (id="upsampling", label="Upsampling", group="Sources & targets", base=false, parent="stokes3d", schemes=AREA,
     about="upsampling = AreaUpsampling(ad, 12): the fast sum over a 12 × 12 rule on every panel, the density interpolated to it, and fewer near corrections",
     run=:(Evaluator(Stokeslet3D(), ad; upsampling=AreaUpsampling(ad, 12))(F))),
    (id="targetdep", label="Kernel K(i, r)", group="Sources & targets", base=false, parent="mh",
     schemes=["mh", "callable", "line"],
     about="a kernel that reads its target i: MHNormalKernel(3.0, bd.n), the normal derivative at the boundary nodes, from area sources (and a function (i, r) -> …, or TractionKernel on line sources)",
     run=:(Evaluator(MHNormalKernel(3.0, bd.n), ad; targets=:boundary, bd)(g))),
    # fast sums
    (id="fmm", label="fast = :fmm", group="Fast sum", base=false, parent="stokes3d",
     schemes=["stokes3d", "laplace3d", "laplace2d", "biharmonic", "mh"],
     about="fast = :fmm: Stokeslet3D summed by FMM3D's Stokes FMM instead of Ewald (the other built-in kernels are summed by their FMM anyway)",
     run=:(Evaluator(Stokeslet3D(), ad; fast=:fmm)(F))),
    (id="direct", label="fast = :direct", group="Fast sum", base=false, parent="stokes3d", schemes=ALL,
     about="fast = :direct: the direct sum over all pairs, threaded over the targets",
     run=:(Evaluator(Stokeslet3D(), ad; fast=:direct)(F))),
    (id="remexp", label="Remainder, exponential", group="Fast sum", base=false, parent="stokes3d",
     schemes=["stokes3d", "laplace3d", "biharmonic"],
     about="remainder = a SymbolKernel whose remainder decays exponentially in k (mobilities 1/k and 1/(2k) + e^{−k}/k over Stokeslet3D), added in Ewald's long-range multiplier; LaplaceKernel3D and BiharmonicGradient2D sum theirs locally",
     run=:(Evaluator(Stokeslet3D(), ad; remainder=SymbolKernel(k -> (1 / k, 1 / (2k) + exp(-k) / k); base=Stokeslet3D(), scale=0.5))(F))),
    (id="remalg", label="Remainder, algebraic", group="Fast sum", base=false, parent="laplace3d",
     schemes=["laplace3d", "biharmonic"],
     about="remainder = the screened kernel e^{−4ρ}/(4πρ) as a SymbolKernel over LaplaceKernel3D (symbol 1/(2√(k² + 16))), whose remainder decays algebraically (its table built to tol = 1e-3, to keep the run short): a local sum of base + remainder, and near corrections of the whole kernel",
     run=:(Evaluator(LaplaceKernel3D(), ad; remainder=SymbolKernel(k -> 1 / (2sqrt(k^2 + 16)); base=LaplaceKernel3D(),
                     scale=0.25, outputs=(:value,), remainder=k -> (s = sqrt(k^2 + 16); -16 / (2k * s * (k + s))), tol=1e-3))(g))),
    # operators
    (id="matrix", label="evaluator_matrix", group="Operators", base=false, parent="stokes3d", schemes=ALL,
     about="evaluator_matrix(E): the dense matrix of an Evaluator, smooth rule plus corrections (on the smaller mesh, an ellipse at p = 6 with one edge layer)",
     run=:(evaluator_matrix(Evaluator(Stokeslet3D(), ads)))),
    (id="mul", label="mul! and *", group="Operators", base=false, parent="stokes3d", schemes=ALL,
     about="size(E), mul!(y, E, x), the five-argument mul! and E * x on interleaved Float64 vectors: an Evaluator as the linear operator of a Krylov solver",
     run=:(let E = Evaluator(Stokeslet3D(), ad), x = interleave(F), y = zeros(size(E, 1))
               mul!(y, E, x); mul!(y, E, x, 2.0, -1.0); E * x end)),
    (id="blocks", label="Local blocks", group="Operators", base=false, parent="stokes3d",
     schemes=["stokes3d", "laplace3d", "laplace2d", "mh", "callable"],
     about="panel_groups(ad) and local_blocks(E, groups): the dense diagonal blocks on the edge columns and the interior panels, for a block-Jacobi preconditioner (targets = :area, a scalar or tensor kernel; on the smaller mesh)",
     run=:(let E = Evaluator(Stokeslet3D(), ads)
               local_blocks(E, panel_groups(ads)); panel_groups(ads; group=false) end)),
    # geometry
    (id="hole", label="A hole", group="Geometry", base=false, parent="stokes3d", schemes=ALL,
     about="discretize_curves of the disc and an off-center hole (reverse_curve makes it clockwise), film_drops, build_mesh of the drop, ncurves, curve_panels, curve_nodes, and an apply on that mesh",
     run=:(let cs = discretize_curves([disc, θ -> Vec2(0.3 + 0.3cos(θ), 0.3sin(θ))], 8)
               bd2, ad2, _ = build_mesh(film_drops(cs[1]; holes=[reverse_curve(cs[2])]), 8)
               ncurves(bd2); curve_panels(bd2, 2); curve_nodes(bd2, 2)
               Evaluator(Stokeslet3D(), ad2)([Vec2(1, 0) for _ in ad2.x]) end)),
    (id="drops", label="Several drops", group="Geometry", base=false, parent="stokes3d", schemes=ALL,
     about="two discs as two drops: build_mesh of a list of curves, ndrops, drop_panels, drop_nodes, panel_drop, node_drop, drop_curves, and an apply",
     run=:(let cs = [discretize_curve(θ -> Vec2(c + 0.8cos(θ), 0.8sin(θ)), 8) for c in (-1.0, 1.0)]
               bd2, ad2, _ = build_mesh(cs, 8)
               ndrops(ad2); drop_panels(ad2, 2); drop_nodes(ad2, 2); panel_drop(ad2); node_drop(ad2); drop_curves(bd2, 2)
               Evaluator(Stokeslet3D(), ad2)([Vec2(1, 0) for _ in ad2.x]) end)),
    (id="layers", label="One edge layer", group="Geometry", base=false, parent="stokes3d", schemes=ALL,
     about="build_mesh(…; edge_layers = 1): one weighted layer on Γ instead of three graded ones, and an apply",
     run=:(let (bd2, ad2, _) = build_mesh(pans, 8; edge_layers=1)
               Evaluator(Stokeslet3D(), ad2)([Vec2(1, 0) for _ in ad2.x]) end)),
    (id="coarse", label="Coarse interior", group="Geometry", base=false, parent="stokes3d", schemes=ALL,
     about="build_mesh(…; coarse_interior = true): every other inner vertex of the edge strip dropped from the interior mesh, and an apply",
     run=:(let (bd2, ad2, _) = build_mesh(pans, 8; coarse_interior=true)
               Evaluator(Stokeslet3D(), ad2)([Vec2(1, 0) for _ in ad2.x]) end)),
    (id="meshutil", label="Mesh utilities", group="Geometry", base=false, parent="stokes3d", schemes=ALL,
     about="mesh_quality, local_feature_size, curve_area, curve_length, and interpolate_nodal of a density at points",
     run=:(begin mesh_quality(ad); local_feature_size(pans); curve_area(pans); curve_length(pans)
                 interpolate_nodal(ad, g, pts) end)),
    # boundary quadrature
    (id="slp2d", label="Boundary single layer", group="Boundary", base=false, parent="line", schemes=["line"],
     about="LineQuad(bd) and the 2D Stokes single layer of a boundary density: slp2d_velocity_matrix on Γ, slp2d_correction to the area nodes, slp2d_eval at points",
     run=:(let lq = LineQuad(bd)
               slp2d_velocity_matrix(lq); slp2d_correction(lq, ad.x); slp2d_eval(lq, pts, ν) end)),
    (id="traction", label="Boundary traction", group="Boundary", base=false, parent="line", schemes=["line"],
     about="the traction of the 2D Stokes single layer: traction_self_matrix on Γ, slp2d_traction_eval at a point, and an Evaluator of TractionKernel (a kernel K(i, r) of the target normals) at points",
     run=:(let lq = LineQuad(bd), n = [normalize(x) for x in pts]
               traction_self_matrix(bd); slp2d_traction_eval(lq, pts[1], n[1], ν)
               Evaluator(OpenSurfaces.TractionKernel(n), bd; targets=pts)(ν) end)),
    # building blocks
    (id="areacorr", label="area_correction", group="Building blocks", base=false, parent="stokes3d", schemes=AREA,
     about="area_correction(ad, K) on its own, the sparse near corrections (here of the Stokeslet at the boundary nodes), with area_local_weights and panel_preimage",
     run=:(let K = OpenSurfaces.StokesKernel(Inf, 1.0)
               area_correction(ad, K; targets=bd.x, target_info=boundary_target_info(ad, bd))
               area_local_weights(ad, 1, ad.x[1], K); panel_preimage(ad.panels[1], ad.x[1]) end)),
    (id="smoothconv", label="SmoothConvolution", group="Building blocks", base=false, parent="stokes3d", schemes=String[],
     about="SmoothConvolution and smooth_apply: a kernel smooth on a given length (a Gaussian) summed over all pairs by NUFFT, as a remainder alone is",
     run=:(smooth_apply(SmoothConvolution(r -> (exp(-4dot(r, r)),), 1, 1, ad.x; scale=0.5), g))),
    (id="filmewald", label="FilmEwald", group="Building blocks", base=false, parent="stokes3d", schemes=["stokes3d"],
     about="FilmEwald(ad.x, bd.x, bd.n; boundary3d = true): the 2D and 3D Stokeslet sums of area and boundary sources on one grid; film_ewald_apply (velocity at the area nodes, traction on Γ), ewald_velocity_boundary and ewald_velocity3d_boundary",
     run=:(let E = FilmEwald(ad.x, bd.x, bd.n; μ_2D=1.0, μ_3D=1.0, boundary3d=true), fa = F .* ad.w, fb = ν .* bd.w
               film_ewald_apply(E, fa, fb); OpenSurfaces.ewald_velocity_boundary(E, fb); ewald_velocity3d_boundary(E, ewald_spread(E, fa), fa) end)),
    (id="volume", label="VolumePotentials", group="Building blocks", base=false, parent="laplace2d",
     schemes=["laplace2d", "biharmonic"],
     about="VolumePotentials(ad, bd; potential = true): V_Ω, ∇V_Ω and ∇B_Ω,κ at the area and boundary nodes (Evaluators of the 2D kernels); volume_grad, volume_potential and volume_bgrad",
     run=:(let VP = VolumePotentials(ad, bd; potential=true)
               volume_grad(VP, g); volume_potential(VP, g); volume_bgrad(VP, g) end)),
    (id="singlelayer", label="SingleLayer3D", group="Building blocks", base=false, parent="laplace3d", schemes=["laplace3d"],
     about="SingleLayer3D(ad) and single_layer: the 3D Laplace single layer of a smooth density (an Evaluator of LaplaceKernel3D)",
     run=:(single_layer(SingleLayer3D(ad), g))),
    (id="fastsums", label="Fast sums alone", group="Building blocks", base=false, parent="stokes3d", schemes=["stokes3d"],
     about="the fast sums on their own: stokes2d_apply (with tractions), stokes3d_plane_apply, EwaldStokes3D and ewald_apply, and the direct versions stokes2d_direct and stokes3d_plane_direct",
     run=:(let n = [normalize(x) for x in pts]
               stokes2d_apply(ad.x, F, pts; traction_normals=n); stokes3d_plane_apply(ad.x, F)
               ewald_apply(EwaldStokes3D(ad.x), F)
               stokes2d_direct(ad.x, F, pts; traction_normals=n); stokes3d_plane_direct(ad.x, F, pts) end)),
    (id="symbolkernel", label="SymbolKernel as kernel", group="Building blocks", base=false, parent="callable", schemes=["callable"],
     about="a SymbolKernel as the kernel itself (a tensor one over Stokeslet3D), summed directly at 40 points, with kernel_remainder, table_range and cached_kernel",
     run=:(let K = cached_kernel(() -> SymbolKernel(k -> (1 / k, 1 / (2k) + exp(-k) / k); base=Stokeslet3D(), scale=0.5), (:codemap, 1))
               kernel_remainder(K, 0.3); table_range(K)
               Evaluator(K, ad; targets=pts, fast=:direct)(F) end)),
]

# ---- child: run the settings under coverage, one LCOV tracefile each -------------------------------------------
if Base.JLOptions().code_coverage != 0
    outdir = ARGS[1]
    Core.eval(Main, SETUP)
    for X in SETTINGS
        sec = @elapsed begin
            ccall(:jl_clear_coverage_data, Cvoid, ())
            Core.eval(Main, X.run)
            ccall(:jl_write_coverage_data, Cvoid, (Cstring,), joinpath(outdir, X.id * ".info"))
        end
        println(rpad(X.id, 14), round(sec; digits=1), " s"); flush(stdout)
    end
    exit()
end

# ---- parent: the top-level functions of src/ and their line ranges --------------------------------------------
# A function is a name in a file (all its methods there); its lines run from its definition to the next top-level
# statement, so a docstring belongs to the function it documents.
_unwrap(ex) = ex isa Expr && ex.head === :macrocall ? _unwrap(ex.args[end]) : ex
_docstring(ex) = ex isa Expr && ex.head === :macrocall && ex.args[1] == GlobalRef(Core, Symbol("@doc")) &&
                 ex.args[3] isa String ? ex.args[3] : nothing
const GENERIC = Set([:size, :eltype, :length, :show, :getindex, :iterate, :copy, :convert, :(==), :(*), :mul!])
const BASETYPES = Set([:AbstractVector, :AbstractMatrix, :AbstractArray, :Vector, :Matrix, :Number, :Real, :Integer,
                       :Int, :Float64, :IO, :MIME, :Function, :Any, :Tuple, :NamedTuple])
_tname(t) = t isa Symbol ? t : t isa Expr && t.head === :curly ? _tname(t.args[1]) :
            t isa Expr && t.head === :. ? t.args[end].value : nothing
function _argtype(sig)
    for a in sig.args[2:end]
        a isa Expr && a.head === :parameters && continue
        a isa Expr && a.head === :kw && (a = a.args[1])
        a isa Expr && a.head === :(::) || continue
        t = _tname(a.args[end])
        t === nothing || t in BASETYPES || return t
    end
    nothing
end
function _fname(ex)
    ex = _unwrap(ex)
    ex isa Expr && ex.head in (:function, :(=)) || return nothing
    sig = ex.args[1]
    while sig isa Expr && sig.head in (:where, :(::)) && !(sig.head === :(::) && length(sig.args) == 1)
        sig = sig.args[1]
    end
    sig isa Expr && sig.head === :call || return nothing
    f = sig.args[1]
    f isa Expr && f.head === :curly && (f = f.args[1])
    f isa Expr && f.head === :. && (f = f.args[end].value)                          # LinearAlgebra.mul!, Base.size
    f isa Expr && f.head === :(::) && return "(::" * string(_tname(f.args[end])) * ")"   # (K::MHKernel)(r)
    f isa Symbol || return nothing
    t = f in GENERIC ? _argtype(sig) : nothing
    t === nothing ? string(f) : string(f, "(::", t, ")")
end
function _summary(doc)
    doc === nothing && return ""
    lines = split(rstrip(doc), '\n')              # not strip: the signature block's indent marks it
    k = findfirst(l -> !startswith(l, "    ") && !isempty(strip(l)), lines)       # skip the signature block
    k === nothing && return ""
    para = String[]
    for l in lines[k:end]
        isempty(strip(l)) && break
        push!(para, strip(l))
    end
    X = join(para, " ")
    m = match(r"^(.*?\.)(\s|$)", X)
    m === nothing ? X : m.captures[1]
end
function functions_of(path)
    text = read(path, String)
    nlines = count(==('\n'), text) + 1
    top = Meta.parseall(text; filename=path)
    stmts = Tuple{Int,Any}[]
    line = 1
    for a in top.args
        if a isa LineNumberNode
            line = a.line
        else
            push!(stmts, (line, a))
        end
    end
    fns = Dict{String,Any}()
    for (k, (l0, ex)) in enumerate(stmts)
        name = _fname(ex)
        name === nothing && continue
        l1 = k < length(stmts) ? stmts[k+1][1] - 1 : nlines
        f = get!(fns, name) do
            Dict("name" => name, "line" => l0, "ranges" => UnitRange{Int}[], "doc" => "")
        end
        push!(f["ranges"], l0:l1)
        isempty(f["doc"]) && (f["doc"] = _summary(_docstring(ex)))
    end
    sort!(collect(values(fns)); by=f -> f["line"]), nlines
end
function executed(info)                              # executed lines of an LCOV tracefile: file => Set(lines)
    hits = Dict{String,Set{Int}}()
    file = ""
    for l in eachline(info)
        if startswith(l, "SF:")
            file = realpath(l[4:end])
        elseif startswith(l, "DA:")
            ln, c = parse.(Int, split(l[4:end], ',')[1:2])
            c > 0 && push!(get!(hits, file, Set{Int}()), ln)
        end
    end
    hits
end

# the code of src/ except the module file (docstring, exports, includes, __init__)
files = sort([relpath(joinpath(d, f), SRC) for (d, _, fs) in walkdir(SRC) for f in fs if endswith(f, ".jl") && f != "OpenSurfaces.jl"])
tmp = isempty(ARGS) ? mktempdir() : mkpath(ARGS[1])
if !all(X -> isfile(joinpath(tmp, X.id * ".info")), SETTINGS)
    cmd = `$(Base.julia_cmd()) -t 1 --project=$ROOT --code-coverage=@$SRC --code-coverage=$(joinpath(tmp, "exit.info")) $(@__FILE__) $tmp`
    println("running ", length(SETTINGS), " settings under coverage …")
    run(cmd)
end
hits = Dict(X.id => executed(joinpath(tmp, X.id * ".info")) for X in SETTINGS)
fnlist = Any[]; filelist = Any[]
for f in files
    path = realpath(joinpath(SRC, f))
    fns, nlines = functions_of(path)
    push!(filelist, (path=f, dir=dirname(f) == "" ? "src" : dirname(f), lines=nlines))
    for fn in fns
        used = [X.id for X in SETTINGS if any(r -> any(in(get(hits[X.id], path, Set{Int}())), r), fn["ranges"])]
        push!(fnlist, (name=fn["name"], file=f, line=fn["line"], doc=fn["doc"], used=used))
    end
end

# ---- write the data ---------------------------------------------------------------------------------------------
js(X::AbstractString) = "\"" * escape_string(X) * "\""
js(v::AbstractVector) = "[" * join(js.(v), ",") * "]"
js(x::Real) = string(x)
js(b::Bool) = string(b)
js(nt::NamedTuple) = "{" * join((string(k) * ":" * js(v) for (k, v) in pairs(nt)), ",") * "}"
open(OUT, "w") do io
    println(io, "// Generated by docs/codemap.jl: the top-level functions of OpenSurfaces' src/ and the settings that run them.")
    println(io, "window.CODEMAP = {")
    rev = try strip(read(`git -C $ROOT rev-parse --short HEAD`, String)) catch; "?" end
    println(io, "  generated: ", js(string(Base.Libc.strftime("%Y-%m-%d", time()), ", src at commit ", rev, " (with uncommitted changes, if any)")), ",")
    println(io, "  settings: [")
    for X in SETTINGS
        println(io, "    ", js((id=X.id, label=X.label, group=X.group, about=X.about, base=X.base, parent=X.parent, schemes=X.schemes)), ",")
    end
    println(io, "  ],\n  files: [")
    for f in filelist
        println(io, "    ", js(f), ",")
    end
    println(io, "  ],\n  functions: [")
    for f in fnlist
        println(io, "    ", js(f), ",")
    end
    println(io, "  ],\n};")
end
println("wrote ", relpath(OUT, ROOT), ": ", length(fnlist), " functions in ", length(filelist), " files, ",
        count(f -> !isempty(f.used), fnlist), " run by at least one setting")
