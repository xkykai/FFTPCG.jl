using Oceananigans
using JLD2
using CUDA
using ArgParse

include("benchmark_utils.jl")
include("rough_rayleigh_benard_setup.jl")

function parse_commandline()
    s = ArgParseSettings()

    @add_arg_table! s begin
      "--grid"
        help = "Grid type: one of $(join(GRID_TYPES, ", "))"
        default = "isotropic"
        range_tester = in(GRID_TYPES)
      "--Lx"
        help = "Domain length in x to benchmark, defaulting to the whole sweep"
        arg_type = Int
      "--preconditioners"
        help = "Comma-separated list drawn from $(join(PRECONDITIONERS, ", "))"
        default = join(PRECONDITIONERS, ',')
    end
    return parse_args(s)
end

args = parse_commandline()
grid_type = args["grid"]
preconditioners = split(args["preconditioners"], ',')

arch = GPU()

N = grid_type == "isotropic" ? 32 : 16

sweep_Lxs = 2 .^ (0:12)
Lxs = isnothing(args["Lx"]) ? sweep_Lxs : [args["Lx"]]

warmup_nsteps = 50
nsteps = 50

mkpath("./reports/")
FILE_PATH = joinpath("./reports/", "single_H100$(output_suffix(grid_type)).jld2")

function key_exists(file_path, key)
    isfile(file_path) || return false
    return jldopen(file_path, "r") do file
        haskey(file, key)
    end
end

for Lx in Lxs, precond_name in preconditioners
    if key_exists(FILE_PATH, "Lx$(Lx)/times/$(precond_name)")
        @info "Skipping $precond_name for Lx=$Lx (already benchmarked)"
        continue
    end
    @info "Benchmarking $precond_name for Lx=$Lx"

    grid = setup_grid(arch, N, grid_type; Lx)
    model = setup_model(grid, build_solver(grid, precond_name))

    results = benchmark_time_steps!(model, stable_timestep(grid), nsteps; warmup=warmup_nsteps)
    save_benchmark!(FILE_PATH, results, precond_name; prefix="Lx$(Lx)/")

    grid = nothing
    model = nothing
    GC.gc()
    CUDA.reclaim()
end
