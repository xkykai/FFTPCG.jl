using JLD2
using Statistics

include("single_benchmark_medians.jl")

"""
    scaling_benchmark_means(file_path)

GPU counts `ngpus`, the grid size on each GPU `local_sizes`, and the mean wall time [s] and CG
iterations per timed step on rank 0 of every preconditioner in the scaling benchmark `file_path`,
ordered by GPU count.
"""
function scaling_benchmark_means(file_path)
    jldopen(file_path, "r") do file
        ngpus = sort(parse.(Int, replace.(keys(file), "gpu" => "")))
        rank_0(group, name) = [file["$(n)gpu/rank_0/$group/$name"] for n in ngpus]
        local_sizes = rank_0("size", "FFT")

        time = Dict(name => [mean(t.time for t in stats) for stats in rank_0("times", name)]
                    for (name, _) in PRECONDITIONER_LABELS)
        iterations = Dict(name => mean.(rank_0("cg_iters", name)) for (name, _) in PRECONDITIONER_LABELS)

        return (; ngpus, local_sizes, time, iterations)
    end
end
