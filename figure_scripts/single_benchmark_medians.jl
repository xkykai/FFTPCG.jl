using JLD2
using Statistics

const PRECONDITIONER_LABELS = ["no" => "No preconditioner",
                               "FFT64" => "FFT64 preconditioner",
                               "FFT32" => "FFT32 preconditioner",
                               "DiagonallyDominant" => "Diagonally-dominant preconditioner (MITgcm)",
                               "ColumnwiseTridiagonal" => "Columnwise tridiagonal preconditioner (MITgcm)",
                               "FFT" => "FFT only"]

"""
    single_benchmark_medians(grid, layout)

Total grid points `N` and the median wall time [s] and CG iterations per time step of every
preconditioner benchmarked on the `layout` (`"square"` or `"strip"`) domains of the `grid`
(`"isotropic"`, `"anisotropic"` or `"stretched"`), ordered by `N`, with the strip width `Ny`
and vertical points `Nz` of that grid. Preconditioners missing at a size are `NaN`.
"""
function single_benchmark_medians(grid, layout)
    file_path = grid == "isotropic" ? "./reports/single_H100.jld2" : "./reports/single_H100_$grid.jld2"
    strip_width, Nz = grid == "isotropic" ? (32, 640) : (16, 2560)

    jldopen(file_path, "r") do file
        Nxs = sort(parse.(Int, keys(file[layout])))
        Ny = layout == "square" ? Nxs : strip_width
        N = Nxs .* Ny .* Nz

        medians(group, f) = Dict(name => [haskey(file, "$layout/$Nx/$group/$name") ?
                                          median(f, file["$layout/$Nx/$group/$name"]) : NaN for Nx in Nxs]
                                 for (name, _) in PRECONDITIONER_LABELS)

        return (; N, time=medians("times", t -> t.time), iterations=medians("cg_iters", identity), strip_width, Nz)
    end
end
