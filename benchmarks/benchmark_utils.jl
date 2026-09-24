using CUDA
using MPI
using JLD2
using Oceananigans
using Oceananigans.DistributedComputations: Distributed, Partition, Equal
using Oceananigans.Solvers: ConjugateGradientPoissonSolver

communicator(arch::Distributed) = arch.communicator
communicator(arch) = nothing

solver_iterations(solver::ConjugateGradientPoissonSolver) = solver.conjugate_gradient_solver.iteration
solver_iterations(solver) = 0

gpu_state() = let dev = CUDA.NVML.Device(CUDA.uuid(CUDA.device()))
    (sm_clock = CUDA.NVML.clock_info(dev).sm,
     temperature = CUDA.NVML.temperature(dev),
     power = CUDA.NVML.power_usage(dev))
end

gpu_model() = match(r"[A-Z]\d{2,3}", CUDA.name(CUDA.device())).match

"""
    gpu_block_size(grid_type)

Points `(Nx, Ny, Nz)` that nearly fill one GPU: `N × N × N` for `isotropic` grids and
`N/2 × N/2 × 4N` for `anisotropic` and `stretched` grids, with `N = 640` on GPUs with 80 GB of
memory and `N = 512` otherwise.
"""
function gpu_block_size(grid_type)
    N = CUDA.totalmem(CUDA.device()) > 60e9 ? 640 : 512
    return grid_type == "isotropic" ? (N, N, N) : (N ÷ 2, N ÷ 2, 4N)
end

function benchmark_architecture()
    MPI.Comm_size(MPI.COMM_WORLD) == 1 && return GPU()
    return Distributed(GPU(); partition = Partition(x = Equal()), synchronized_communication = false)
end

"""
    benchmark_time_steps!(model, Δt, nsteps; warmup)

Time `nsteps` calls to `time_step!`, returning the per-step `@timed` `stats`, the pressure
solver `iterations` per step, and the `elapsed` time of the whole loop measured between rank
barriers.

The GPU is synchronized inside the timed block, so `stats[n].time` is the time to complete
the step rather than the time to queue it.
"""
function benchmark_time_steps!(model, Δt, nsteps; warmup = nsteps)
    comm = communicator(model.architecture)
    barrier() = isnothing(comm) || MPI.Barrier(comm)

    for _ in 1:warmup
        time_step!(model, Δt)
    end

    stats = []
    iterations = Int[]

    CUDA.synchronize()
    barrier()
    initial_state = gpu_state()
    t₀ = time_ns()

    for _ in 1:nsteps
        t = @timed begin
            time_step!(model, Δt)
            CUDA.synchronize()
        end
        push!(stats, t)
        push!(iterations, solver_iterations(model.pressure_solver))
    end

    barrier()
    elapsed = (time_ns() - t₀) * 1e-9

    return (; stats, iterations, elapsed, initial_state, final_state = gpu_state())
end

"""
    save_benchmark!(file_path, results, name; prefix="")

Write the timing `results` for preconditioner `name` into `file_path`, replacing any existing
entry, under the keys `prefix * "times/" * name` and its `cg_iters` and `gpu_state` siblings.
"""
function save_benchmark!(file_path, results, name; prefix = "")
    jldopen(file_path, "a") do file
        for group in ("times", "cg_iters", "gpu_state")
            key = "$prefix$group/$name"
            haskey(file, key) && delete!(file, key)
        end

        file["$(prefix)times/$name"] = results.stats
        file["$(prefix)cg_iters/$name"] = results.iterations
        file["$(prefix)gpu_state/$name"] = (initial = results.initial_state,
                                            final = results.final_state,
                                            elapsed = results.elapsed)
    end
end
