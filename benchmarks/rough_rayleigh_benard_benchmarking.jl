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
      "--layout"
        help = "Horizontal layout: square (n × n roughness periods) or strip (n² × 1), defaulting to both"
        range_tester = in(("square", "strip"))
      "--bumps"
        help = "Roughness periods along each side of the square, defaulting to the whole sweep"
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

N = 128 # 16 points per roughness period

sweep_bumps = grid_type == "isotropic" ? 2 .^ (0:6) : 2 .^ (0:4)
bumps = isnothing(args["bumps"]) ? sweep_bumps : [args["bumps"]]
layouts = isnothing(args["layout"]) ? ("square", "strip") : (args["layout"],)

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

for layout in layouts, n in bumps, precond_name in preconditioners
    if key_exists(FILE_PATH, "$layout/$n/times/$(precond_name)")
        @info "Skipping $precond_name for the $layout with n = $n (already benchmarked)"
        continue
    end
    @info "Benchmarking $precond_name for the $layout with n = $n"

    Nx, Ny = layout == "square" ? (16n, 16n) : (16n^2, 16)
    grid = setup_grid(arch, N, grid_type; Lx = Nx / N, Ly = Ny / N)
    model = setup_model(grid, build_solver(grid, precond_name))

    results = benchmark_time_steps!(model, stable_timestep(grid), nsteps; warmup=warmup_nsteps)
    save_benchmark!(FILE_PATH, results, precond_name; prefix="$layout/$n/")

    grid = nothing
    model = nothing
    GC.gc()
    CUDA.reclaim()
end
