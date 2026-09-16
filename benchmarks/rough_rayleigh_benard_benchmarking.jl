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

Ns = grid_type == "isotropic" ? [32, 64, 96, 128, 192, 256, 384, 512] : [16, 32, 64, 96, 128, 192, 256]

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

for N in Ns, precond_name in preconditioners
    if key_exists(FILE_PATH, "$(N)/times/$(precond_name)")
        @info "Skipping $precond_name for N=$N (already benchmarked)"
        continue
    end
    @info "Benchmarking $precond_name for N=$N"

    grid = setup_grid(arch, N, grid_type)
    model = setup_model(grid, build_solver(grid, precond_name))

    results = benchmark_time_steps!(model, stable_timestep(grid), nsteps; warmup=warmup_nsteps)
    save_benchmark!(FILE_PATH, results, precond_name; prefix="$(N)/")

    grid = nothing
    model = nothing
    GC.gc()
    CUDA.reclaim()
end
