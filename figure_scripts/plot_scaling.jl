using CairoMakie
using Makie

include("scaling_benchmark_means.jl")

const GRIDS = ["isotropic" => "", "anisotropic" => "_anisotropic", "stretched" => "_stretched"]

#%%
colors = Makie.wong_colors();
linewidth = 4
markersize = 13

for kind in ("weak", "strong")
    fig = Figure(size=(1500, 1300), fontsize=20)

    for (row, (grid, suffix)) in enumerate(GRIDS)
        data = scaling_benchmark_means("./reports/$(kind)scaling_A100$suffix.jld2")
        points = join(data.local_sizes[1], " × ") * (kind == "weak" ? " per GPU" : " in total")
        axis_kw = (; xlabel="GPUs", xscale=log2, xticks=data.ngpus, title="$grid, $points")
        axtime = Axis(fig[row, 1]; axis_kw..., ylabel="Wall time per timestep (s)", yscale=log10)
        axiters = Axis(fig[row, 2]; axis_kw..., ylabel="CG iters per timestep", yscale=log10)
        axefficiency = Axis(fig[row, 3]; axis_kw..., ylabel="$(titlecase(kind)) scaling efficiency")
        hlines!(axefficiency, 1; color=:gray, linestyle=:dash)

        for (i, (name, label)) in enumerate(PRECONDITIONER_LABELS)
            color = colors[i]
            linestyle = name == "FFT" ? :dot : :solid
            time = data.time[name]
            efficiency = kind == "weak" ? time[1] ./ time : time[1] ./ (data.ngpus .* time)

            scatterlines!(axtime, data.ngpus, time; label, linewidth, color, markersize, linestyle)
            scatterlines!(axefficiency, data.ngpus, efficiency; linewidth, color, markersize, linestyle)
            name == "FFT" && continue
            scatterlines!(axiters, data.ngpus, data.iterations[name]; linewidth, color, markersize)
        end
    end

    Legend(fig[length(GRIDS) + 1, :], content(fig[1, 1]), nbanks=3, orientation=:horizontal, patchsize=(40, 20))
    Label(fig[0, :], "$(titlecase(kind)) scaling benchmark (NVIDIA A100)", font=:bold)

    display(fig)
    save("./Output/$(kind)scaling_A100.png", fig, px_per_unit=2)
end
#%%
