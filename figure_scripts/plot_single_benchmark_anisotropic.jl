using JLD2
using Statistics
using CairoMakie
using Makie

filepath = "./reports/single_H100_anisotropic.jld2"

Ns = [16, 32, 64, 96, 128, 192, 256]

file = jldopen(filepath, "r")

precond_names = ["no", "FFT64", "FFT32", "DiagonallyDominant", "ColumnwiseTridiagonal"]

median_times = Dict{String, Vector{Float64}}()
median_cg_iters = Dict{String, Vector{Float64}}()

for name in precond_names
    median_times[name] = zeros(length(Ns))
    median_cg_iters[name] = zeros(length(Ns))
    for (i, N) in enumerate(Ns)
        nsamples = length(file["$(N)/times/$(name)"])
        times = file["$(N)/times/$(name)"]
        median_times[name][i] = median([t.time for t in times])
        cg_iters = file["$(N)/cg_iters/$(name)"]
        median_cg_iters[name][i] = median(cg_iters)
    end
end

median_times["FFT_only"] = zeros(length(Ns))
for (i, N) in enumerate(Ns)
    nsamples = length(file["$(N)/times/FFTstep"])
    times = file["$(N)/times/FFTstep"]
    median_times["FFT_only"][i] = median([t.time for t in times])
end

median_cg_costs = Dict{String, Vector{Float64}}()
for name in precond_names
    median_cg_costs[name] = zeros(length(Ns))
     for (i, N) in enumerate(Ns)
         times = [t.time for t in file["$(N)/times/$(name)"]]
         cg_iters = file["$(N)/cg_iters/$(name)"]
         median_FFT_time = median_times["FFT_only"][i]
        #  median_cg_costs[name][i] = median(times ./ cg_iters) / median_FFT_time
         median_cg_costs[name][i] = median(times) / median_FFT_time
     end
end

#%%
colors = Makie.wong_colors();
linewidth = 5
markersize = 15
fig = Figure(size=(1300, 550), fontsize=23)
axtime = Axis(fig[1, 1], xlabel="N (Problem size is N² × 8N)", ylabel="Wall time per timestep (s)", yscale = log10, xscale=log2)
axiters = Axis(fig[1, 2], xlabel="N (Problem size is N² × 8N)", ylabel="CG iters per timestep", yscale = log10, xscale=log2)
axratio = Axis(fig[1, 3], xlabel="N (Problem size is N² × 8N)", ylabel="Method time /\nFFT-only solve time", yscale = log10, xscale=log2)

scatterlines!(axtime, Ns, median_times["no"], label="No preconditioner"; linewidth, color=colors[1], markersize)
scatterlines!(axtime, Ns, median_times["FFT64"], label="FFT64 preconditioner"; linewidth, color=colors[2], markersize)
scatterlines!(axtime, Ns, median_times["FFT32"], label="FFT32 preconditioner"; linewidth, color=colors[3], markersize)
scatterlines!(axtime, Ns, median_times["DiagonallyDominant"], label="Diagonally-dominant preconditioner (MITgcm)"; linewidth, color=colors[4], markersize)
scatterlines!(axtime, Ns, median_times["ColumnwiseTridiagonal"], label="Columnwise tridiagonal preconditioner (MITgcm)"; linewidth, color=colors[5], markersize)
scatterlines!(axtime, Ns, median_times["FFT_only"], label="FFT only"; linewidth, color=colors[6], markersize, linestyle=:dot)

scatterlines!(axiters, Ns, median_cg_iters["no"], label="No Preconditioner"; linewidth, color=colors[1], markersize)
scatterlines!(axiters, Ns, median_cg_iters["FFT64"], label="FFT64 Preconditioner"; linewidth, color=colors[2], markersize)
scatterlines!(axiters, Ns, median_cg_iters["FFT32"], label="FFT32 Preconditioner"; linewidth, color=colors[3], markersize)
scatterlines!(axiters, Ns, median_cg_iters["DiagonallyDominant"], label="Diagonally-Dominant Preconditioner (MITgcm)"; linewidth, color=colors[4], markersize)
scatterlines!(axiters, Ns, median_cg_iters["ColumnwiseTridiagonal"], label="Columnwise Tridiagonal Preconditioner (MITgcm)"; linewidth, color=colors[5], markersize)

scatterlines!(axratio, Ns, median_cg_costs["no"], label="No Preconditioner"; linewidth, color=colors[1], markersize)
scatterlines!(axratio, Ns, median_cg_costs["FFT64"], label="FFT64 Preconditioner"; linewidth, color=colors[2], markersize)
scatterlines!(axratio, Ns, median_cg_costs["FFT32"], label="FFT32 Preconditioner"; linewidth, color=colors[3], markersize)
scatterlines!(axratio, Ns, median_cg_costs["DiagonallyDominant"], label="Diagonally-Dominant Preconditioner (MITgcm)"; linewidth, color=colors[4], markersize)
scatterlines!(axratio, Ns, median_cg_costs["ColumnwiseTridiagonal"], label="Columnwise Tridiagonal Preconditioner (MITgcm)"; linewidth, color=colors[5], markersize)

Legend(fig[2, :], axtime, nbanks=3, orientation=:horizontal, patchsize=(40, 20))
Label(fig[0, :], "Single GPU Benchmark (NVIDIA H100)", font=:bold)

display(fig)
save("./Output/benchmark_single_H100_anisotropic_grc.png", fig, px_per_unit=4)
#%%
