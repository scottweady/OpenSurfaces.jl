# Build OpenSurfaces' documentation (from the package root):
#   julia --project=docs -e 'using Pkg; Pkg.instantiate()'     (once; OpenSurfaces is taken by path, docs/Project.toml)
#   julia --project=docs docs/make.jl
# The site is written to docs/build/ (open docs/build/index.html). Figures: docs/figures/ (make_data.jl writes the
# data, plot_figures.py draws them); the code map: docs/codemap.jl. On GitHub, .github/workflows/Documentation.yml builds
# it on every push to main and `deploydocs` publishes it to the gh-pages branch (https://scottweady.github.io/OpenSurfaces.jl/).
using Documenter, OpenSurfaces

makedocs(
    sitename = "OpenSurfaces.jl",
    authors = "Scott Weady",
    modules = [OpenSurfaces],
    format = Documenter.HTML(
        prettyurls = get(ENV, "CI", "false") == "true",
        mathengine = Documenter.KaTeX(),
        assets = ["assets/themed.css", "assets/sectionlinks.js", "assets/codemap/codemap.css", "assets/codemap/codemap.js"],
        collapselevel = 1,
        edit_link = nothing,
        repolink = "https://github.com/scottweady/OpenSurfaces.jl",
        size_threshold = 400 * 1024,
    ),
    pages = [
        "Home" => "index.md",
        "Getting started" => "getting_started.md",
        "Method" => [
            "Overview" => "method/index.md",
            "method/geometry.md",
            "method/quadrature.md",
            "method/fast_sums.md",
            "method/symbols.md",
            "method/limits.md",
        ],
        "Examples" => "examples.md",
        "API reference" => "api.md",
        "Code map" => "codemap.md",
    ],
    checkdocs = :exports,    # every exported name has a docstring in the API reference
    remotes = nothing,       # no source links
)

deploydocs(
    repo = "github.com/scottweady/OpenSurfaces.jl.git",
    devbranch = "main",
    versions = nothing,
)
