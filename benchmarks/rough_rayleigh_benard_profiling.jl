# Profile one weak-scaling configuration with Nsight Systems, one report per rank:
#
#   mpiexec -n 2 nsys profile --trace=cuda,nvtx --capture-range=cudaProfilerApi --capture-range-end=stop \
#       --output=rank_%q{PMI_RANK} julia --project benchmarks/rough_rayleigh_benard_profiling.jl \
#       --grid stretched --preconditioner FFT32
#
# `nsys` comes with the CUDA toolkit (`module load nvhpc cuda` on Derecho). `--capture-range=cudaProfilerApi`
# records only the profiled steps, each in an NVTX range `time step <n>`. Without `nsys` the script only
# prints timings.
#
# - Set `JULIA_CUDA_MEMORY_POOL=none`: Cray MPICH can't share pool-allocated memory between ranks.
# - Don't add `--trace=mpi`: it breaks MPI through MPItrampoline.
# - If `using MPI` fails under `nsys` with an `OPENSSL_3.3.0` error, add
#   `--env-var=LD_PRELOAD=<Julia's lib/julia/libcrypto.so.3>`.

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
        GC.gc(false)
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
