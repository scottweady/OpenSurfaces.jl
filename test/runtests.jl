# OpenSurfaces regression tests (test_*.jl here), at the production settings and against exact or reference
# solutions: kernels, FMM and Ewald sums against direct sums; kernels from Fourier symbols; boundary and area
# quadrature; boundary panels, meshes, holes, several drops; the common evaluator; volume potentials and
# the 3D Laplace single layer on the disc.
# Each file runs in its own module so their helpers do not collide; shared references are in common/. The
# convergence sweeps and timings behind them are the opt-in tier validation/run_all.jl.
# From this package's directory (src/OpenSurfaces):
#   julia -t 8 --project -e 'using Pkg; Pkg.test()'                                       (all files)
#   julia -t 8 --project -e 'using Pkg; Pkg.test(test_args=["kernels", "symbol"])'         (only test_kernels.jl, test_symbol.jl)
using Test, Printf

files = sort(filter(f -> startswith(f, "test_") && endswith(f, ".jl"), readdir(@__DIR__)))
isempty(ARGS) || (files = filter(f -> f[6:end-3] in ARGS, files))

@testset "OpenSurfaces" begin
    for f in files
        t = @elapsed @testset verbose=true "$f" begin
            m = Core.eval(Main, :(module $(Symbol(f[1:end-3])) end))   # defines `include` for nested includes
            Base.include(m, joinpath(@__DIR__, f))
        end
        @printf("  %-30s %7.1f s\n", f, t)
    end
end
