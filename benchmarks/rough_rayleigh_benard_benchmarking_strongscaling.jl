using MPI
using Oceananigans
using JLD2
using CUDA
using ArgParse

include("benchmark_utils.jl")
include("rough_rayleigh_benard_setup.jl")

MPI.Init()

function parse_commandline()
    s = ArgParseSettings()

    @add_arg_table! s begin
      "--ngpus"
        help = "Number of GPUs to use"
        arg_type = Int
        default = 1
      "--grid"
        help = "Grid type: one of $(join(GRID_TYPES, ", "))"
        default = "isotropic"
        range_tester = in(GRID_TYPES)
      "--preconditioners"
        help = "Comma-separated list drawn from $(join(PRECONDITIONERS, ", "))"
        default = join(PRECONDITIONERS, ',')
    end
    return parse_args(s)
end

args = parse_commandline()
ngpus = args["ngpus"]
grid_type = args["grid"]
preconditioners = split(args["preconditioners"], ',')

arch = benchmark_architecture()
local_rank = MPI.Comm_rank(MPI.COMM_WORLD)

N = points_per_unit_length(grid_type)

warmup_nsteps = 50
nsteps = 50

OUTPUT_DIR = "./reports/strongscaling_$(gpu_model())$(output_suffix(grid_type))/benchmark_$(ngpus)gpu"
mkpath(OUTPUT_DIR)
FILE_PATH = joinpath(OUTPUT_DIR, "rank_$(local_rank).jld2")

for precond_name in preconditioners
    @info "Benchmarking $precond_name on rank $local_rank"

    grid = setup_grid(arch, N, grid_type)
    model = setup_model(grid, build_solver(grid, precond_name); seed = 1234 + local_rank)

    results = benchmark_time_steps!(model, stable_timestep(grid), nsteps; warmup=warmup_nsteps)
    save_benchmark!(FILE_PATH, results, precond_name)

    grid = nothing
    model = nothing
    GC.gc()
    CUDA.reclaim()
end
