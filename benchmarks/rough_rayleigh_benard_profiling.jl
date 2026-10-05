t₀ = time()

using MPI
using Oceananigans
using JLD2
using CUDA
using NVTX
using ArgParse
using Printf

include("benchmark_utils.jl")
include("rough_rayleigh_benard_setup.jl")

MPI.Init()

function parse_commandline()
    s = ArgParseSettings()

    @add_arg_table! s begin
      "--grid"
        help = "Grid type: one of $(join(GRID_TYPES, ", "))"
        default = "stretched"
        range_tester = in(GRID_TYPES)
      "--preconditioner"
        help = "One of $(join(PRECONDITIONERS, ", "))"
        default = "FFT64"
        range_tester = in(PRECONDITIONERS)
      "--warmup-steps"
        help = "Time steps taken before profiling starts"
        arg_type = Int
        default = 3
      "--steps"
        help = "Time steps profiled"
        arg_type = Int
        default = 2
      "--maxiter"
        help = "CG iterations per pressure solve, at most"
        arg_type = Int
        default = 100
    end
    return parse_args(s)
end

args = parse_commandline()
grid_type = args["grid"]
precond_name = args["preconditioner"]

comm = MPI.COMM_WORLD
ngpus = MPI.Comm_size(comm)
report(message) = MPI.Comm_rank(comm) == 0 && (@printf("+%8.1f s  %s\n", time() - t₀, message); flush(stdout))
report("packages loaded")

arch = benchmark_architecture()
Nx, Ny, Nz = gpu_block_size(grid_type)
grid = setup_grid(arch, grid_type, ngpus * Nx, Ny, Nz)
MPI.Barrier(comm)
report("grid built: $(ngpus * Nx) × $Ny × $Nz, $grid_type, $ngpus GPUs")

solver = build_solver(grid, precond_name)
isnothing(solver) || (solver.conjugate_gradient_solver.maxiter = args["maxiter"])
model = setup_model(grid, solver; seed = 1234 + MPI.Comm_rank(comm))
Δt = stable_timestep(grid)
MPI.Barrier(comm)
report("model built: preconditioner $precond_name, at most $(args["maxiter"]) CG iterations per solve")

function timed_step!(model, Δt, comm)
    stats = @timed begin
        time_step!(model, Δt)
        CUDA.synchronize()
        MPI.Barrier(comm)
    end
    return @sprintf("%.2f s (%.2f s in %d garbage collections), %d CG iterations in the last solve",
                    stats.time, stats.gctime, stats.gcstats.pause, solver_iterations(model.pressure_solver))
end

for n in 1:args["warmup-steps"]
    report("warmup step $n: " * timed_step!(model, Δt, comm))
end

CUDA.@profile external=true begin
    for n in 1:args["steps"]
        summary = NVTX.@range "time step $n" timed_step!(model, Δt, comm)
        report("profiled step $n: " * summary)
    end
end
report("done")
