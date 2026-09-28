using CairoMakie
using Makie

include("single_benchmark_medians.jl")

file_path = "./reports/single_H100.jld2"
strip_width, Nz = 32, 640

layouts = ["square" => "Square (Nx × Nx × $Nz)", "strip" => "Strip (Nx × $strip_width × $Nz)"]
linestyles = [:solid, :dash]
data = [single_benchmark_medians(file_path, layout; strip_width, Nz) for (layout, _) in layouts]

#%%
colors = Makie.wong_colors();
linewidth = 4
markersize = 13
xlabel = "Grid points"
fig = Figure(size=(1300, 650), fontsize=23)
axtime = Axis(fig[1, 1]; xlabel, ylabel="Wall time per timestep (s)", yscale=log10, xscale=log10)
axiters = Axis(fig[1, 2]; xlabel, ylabel="CG iters per timestep", yscale=log10, xscale=log10)
axratio = Axis(fig[1, 3]; xlabel, ylabel="Method time /\nFFT-only solve time", yscale=log10, xscale=log10)

for (d, linestyle) in zip(data, linestyles), (i, (name, _)) in enumerate(PRECONDITIONER_LABELS)
    color = colors[i]
    scatterlines!(axtime, d.N, d.time[name]; linewidth, color, markersize, linestyle)
    name == "FFT" && continue
    scatterlines!(axiters, d.N, d.iterations[name]; linewidth, color, markersize, linestyle)
    scatterlines!(axratio, d.N, d.time[name] ./ d.time["FFT"]; linewidth, color, markersize, linestyle)
end

preconditioner_entries = [LineElement(; color=colors[i], linewidth) for i in eachindex(PRECONDITIONER_LABELS)]
layout_entries = [LineElement(; color=:black, linestyle, linewidth) for linestyle in linestyles]
Legend(fig[2, :], [preconditioner_entries, layout_entries],
       [last.(PRECONDITIONER_LABELS), last.(layouts)], ["Solver", "Domain"],
       nbanks=3, orientation=:horizontal, patchsize=(40, 20))
Label(fig[0, :], "Single GPU Benchmark (NVIDIA H100): square vs strip domains", font=:bold)

display(fig)
save("./Output/benchmark_single_H100_square_vs_strip.png", fig, px_per_unit=4)
#%%
