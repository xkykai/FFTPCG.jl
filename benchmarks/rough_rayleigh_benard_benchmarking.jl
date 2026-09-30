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
        help = "Layout: square (Nx × Nx points) or strip (Nx × 32 points, Nx × 16 for anisotropic and stretched grids), defaulting to both"
        range_tester = in(("square", "strip"))
      "--nx"
        help = "Points in x, defaulting to the whole sweep"
        arg_type = Int
      "--preconditioners"
        help = "Comma-separated list drawn from $(join(PRECONDITIONERS, ", "))"
        default = join(PRECONDITIONERS, ',')
      "--warmup-steps"
        help = "Time steps taken before timing starts"
        arg_type = Int
        default = 50
      "--steps"
        help = "Time steps timed"
        arg_type = Int
        default = 50
    end
    return parse_args(s)
end

args = parse_commandline()
grid_type = args["grid"]
preconditioners = split(args["preconditioners"], ',')

arch = GPU()

Nx_max, _, Nz = gpu_block_size(grid_type)

if grid_type == "isotropic"
    square_sides = (16, 32, 64, 128, 256, 512, 640)
    strip_lengths, strip_width = (128, 512, 2048, 8192, 12800), 32
else
    square_sides = (16, 32, 64, 128, 256, 320)
    strip_lengths, strip_width = (64, 256, 1024, 4096, 6400), 16
end

sizes = (square = [(Nx, Nx, Nz) for Nx in square_sides if Nx ≤ Nx_max],
         strip = [(Nx, strip_width, Nz) for Nx in strip_lengths])
layouts = isnothing(args["layout"]) ? ("square", "strip") : (args["layout"],)

warmup_nsteps = args["warmup-steps"]
nsteps = args["steps"]

mkpath("./reports/")
FILE_PATH = joinpath("./reports/", "single_$(gpu_model())$(output_suffix(grid_type)).jld2")

for layout in layouts, (Nx, Ny, Nz) in sizes[Symbol(layout)], precond_name in preconditioners
    isnothing(args["nx"]) || Nx == args["nx"] || continue

    if key_exists(FILE_PATH, "$layout/$Nx/times/$(precond_name)")
        @info "Skipping $precond_name for the $layout with Nx = $Nx (already benchmarked)"
        continue
    end
    @info "Benchmarking $precond_name for the $layout with Nx = $Nx"

    grid = setup_grid(arch, grid_type, Nx, Ny, Nz)
    model = setup_model(grid, build_solver(grid, precond_name))

    results = benchmark_time_steps!(model, stable_timestep(grid), nsteps; warmup=warmup_nsteps)
    save_benchmark!(FILE_PATH, results, precond_name; prefix="$layout/$Nx/")

    grid = nothing
    model = nothing
    GC.gc()
    CUDA.reclaim()
end
