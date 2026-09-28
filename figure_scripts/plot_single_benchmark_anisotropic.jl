using CairoMakie
using Makie

include("single_benchmark_medians.jl")

data = single_benchmark_medians("./reports/single_H100_anisotropic.jld2", "square"; strip_width=16, Nz=2560)

#%%
colors = Makie.wong_colors();
linewidth = 5
markersize = 15
xlabel = "Grid points (Nx × Nx × 2560)"
fig = Figure(size=(1300, 550), fontsize=23)
axtime = Axis(fig[1, 1]; xlabel, ylabel="Wall time per timestep (s)", yscale=log10, xscale=log10)
axiters = Axis(fig[1, 2]; xlabel, ylabel="CG iters per timestep", yscale=log10, xscale=log10)
axratio = Axis(fig[1, 3]; xlabel, ylabel="Method time /\nFFT-only solve time", yscale=log10, xscale=log10)

for (i, (name, label)) in enumerate(PRECONDITIONER_LABELS)
    color = colors[i]
    linestyle = name == "FFT" ? :dot : :solid
    scatterlines!(axtime, data.N, data.time[name]; label, linewidth, color, markersize, linestyle)
    name == "FFT" && continue
    scatterlines!(axiters, data.N, data.iterations[name]; linewidth, color, markersize)
    scatterlines!(axratio, data.N, data.time[name] ./ data.time["FFT"]; linewidth, color, markersize)
end

Legend(fig[2, :], axtime, nbanks=3, orientation=:horizontal, patchsize=(40, 20))
Label(fig[0, :], "Single GPU Benchmark (NVIDIA H100)", font=:bold)

display(fig)
save("./Output/benchmark_single_H100_anisotropic_grc.png", fig, px_per_unit=4)
#%%
