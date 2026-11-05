using JLD2
using Statistics

const PRECONDITIONER_LABELS = ["no" => "No preconditioner",
                               "FFT64" => "FFT64 preconditioner",
                               "FFT32" => "FFT32 preconditioner",
                               "DiagonallyDominant" => "Diagonally-dominant preconditioner (MITgcm)",
                               "ColumnwiseTridiagonal" => "Columnwise tridiagonal preconditioner (MITgcm)",
                               "FFT" => "FFT only (inexact solution)"]

"""
    single_benchmark_medians(grid, layout; front=false)

Grid `sizes`, total grid points `N`, and the median wall time [s] and CG iterations per time
step of every preconditioner benchmarked on the `layout` (`"square"` or `"strip"`) domains of
the `grid` (`"isotropic"`, `"anisotropic"` or `"stretched"`), with a buoyancy front if `front`,
ordered by `N`. Preconditioners missing at a size are `NaN`.
"""
function single_benchmark_medians(grid, layout; front=false)
    case = front ? "single_front" : "single"
    suffix = grid == "isotropic" ? "" : "_$grid"
    file_path = "./reports/$(case)_H100$suffix.jld2"

    jldopen(file_path, "r") do file
        Nxs = sort(parse.(Int, keys(file[layout])))
        sizes = [file["$layout/$Nx/size/FFT"] for Nx in Nxs]
        N = prod.(sizes)

        medians(group, f) = Dict(name => [haskey(file, "$layout/$Nx/$group/$name") ?
                                          median(f, file["$layout/$Nx/$group/$name"]) : NaN for Nx in Nxs]
                                 for (name, _) in PRECONDITIONER_LABELS)

        return (; sizes, N, time=medians("times", t -> t.time), iterations=medians("cg_iters", identity))
    end
end
