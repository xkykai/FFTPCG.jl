using CairoMakie
using Makie

include("scaling_benchmark_means.jl")

const GRIDS = ["Isotropic" => "", "Anisotropic" => "_anisotropic", "Stretched" => "_stretched"]

#%%
colors = Makie.wong_colors()
markers = [:circle, :rect, :utriangle, :diamond, :dtriangle, :star5]
linewidth = 1.8
markersize = 8
letters = 'a':'z'
decades = (10.0 .^ (-1:4), [rich("10", superscript(replace(string(n), "-" => "−"))) for n in -1:4])

for kind in ("weak", "strong"), case in ("", "_front")
    fig = Figure(size=(510, 540), fontsize=9, figure_padding=(2, 8, 2, 4))

    ylabels = ["Wall time per time step (s)", "CG iterations per time step", "$(titlecase(kind)) scaling efficiency"]
    axes = Matrix{Axis}(undef, 3, length(GRIDS))

    for (col, (grid, suffix)) in enumerate(GRIDS)
        data = scaling_benchmark_means("./reports/$(kind)scaling$(case)_A100$suffix.jld2")
        points = join(data.local_sizes[1], " × ") * (kind == "weak" ? " per GPU" : " total")
        Label(fig[-1, col], grid; font=:bold, tellwidth=false)
        Label(fig[0, col], points; tellwidth=false)

        axis_kw = (; titlealign=:left, titlesize=9, titlegap=2, xscale=log2, xticks=data.ngpus, xlabel="Number of GPUs",
                     xgridvisible=false, ygridvisible=false, xminorticksvisible=false)
        log_kw = (; yscale=log10, yticks=decades, yminorticks=IntervalsBetween(9), yminorticksvisible=true, yminortickwidth=0.5)
        axtime = Axis(fig[1, col]; axis_kw..., log_kw..., title="($(letters[col]))", ylabel=ylabels[1])
        axiters = Axis(fig[2, col]; axis_kw..., log_kw..., title="($(letters[col + 3]))", ylabel=ylabels[2])
        axefficiency = Axis(fig[3, col]; axis_kw..., title="($(letters[col + 6]))", ylabel=ylabels[3],
                            limits=(nothing, (0, 1.08)), yticks=0:0.2:1)
        hlines!(axefficiency, 1; color=:gray60, linewidth=1, linestyle=:dash)

        for (i, (name, label)) in enumerate(PRECONDITIONER_LABELS)
            kw = (; label, linewidth, markersize, color=colors[i], marker=markers[i], linestyle=name == "FFT" ? :dot : :solid)
            time = data.time[name]
            efficiency = kind == "weak" ? time[1] ./ time : time[1] ./ (data.ngpus .* time)

            scatterlines!(axtime, data.ngpus, time; kw...)
            scatterlines!(axefficiency, data.ngpus, efficiency; kw...)
            name == "FFT" || scatterlines!(axiters, data.ngpus, data.iterations[name]; kw...)
        end

        axes[:, col] = [axtime, axiters, axefficiency]
    end

    linkxaxes!(axes...)
    for row in 1:3
        linkyaxes!(axes[row, :]...)
    end

    for ax in axes[1:end-1, :]
        hidexdecorations!(ax; ticks=false)
    end

    for ax in axes[:, 2:end]
        hideydecorations!(ax; ticks=false, minorticks=false)
    end

    Legend(fig[4, 1:length(GRIDS)], axes[1, 1]; orientation=:horizontal, nbanks=3, tellwidth=false,
           framevisible=false, patchsize=(24, 10), colgap=10, rowgap=0, padding=(0, 0, 0, 0))

    colgap!(fig.layout, 8)
    rowgap!(fig.layout, 6)
    rowgap!(fig.layout, 1, 2)

    display(fig)
    save("./Output/$(kind)scaling$(case)_A100.pdf", fig, pt_per_unit=1)
    save("./Output/$(kind)scaling$(case)_A100.png", fig, px_per_unit=6)
end
#%%
