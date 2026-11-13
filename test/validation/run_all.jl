# OpenSurfaces validation tier: the convergence sweeps and timings behind the regression tests (val_*.jl here).
# Opt-in; the regression tests are test/runtests.jl (`Pkg.test()`). It needs QuadGK besides OpenSurfaces: from the
# repository root, in a temporary environment with both,
#   julia -t 8 -e 'using Pkg; Pkg.activate(temp=true); Pkg.develop(path="."); Pkg.add("QuadGK");
#                  include("test/validation/run_all.jl")'
# (only some files: `push!(ARGS, "area_quad", "fmm")` before the include runs val_area_quad.jl and val_fmm.jl).
using Test, Printf

files = sort(filter(f -> startswith(f, "val_") && endswith(f, ".jl"), readdir(@__DIR__)))
isempty(ARGS) || (files = filter(f -> f[5:end-3] in ARGS, files))

@testset "OpenSurfaces validation" begin
    for f in files
        t = @elapsed @testset "$f" begin
            m = Core.eval(Main, :(module $(Symbol(f[1:end-3])) end))
            Base.include(m, joinpath(@__DIR__, f))
        end
        @printf("%-30s %7.1f s\n", f, t)
    end
end
