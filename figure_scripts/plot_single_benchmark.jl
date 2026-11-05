using CairoMakie
using Makie

include("single_benchmark_medians.jl")

const GRIDS = ["Isotropic" => "isotropic", "Anisotropic" => "anisotropic", "Stretched" => "stretched"]

#%%
colors = Makie.wong_colors()
markers = [:circle, :rect, :utriangle, :diamond, :dtriangle, :star5]
linewidth = 1.8
markersize = 8
letters = 'a':'z'
decades = (10.0 .^ (-3:9), [rich("10", superscript(replace(string(n), "-" => "−"))) for n in -3:9])
powers_of_two = (2.0 .^ (0:2:40), [rich("2", superscript(string(n))) for n in 0:2:40])
domain_styles = [(; linestyle=:solid, linewidth), (; linestyle=:dot, linewidth=1.8linewidth)]
domains = [LineElement(; color=:black, style...) for style in domain_styles]
legend_kw = (; tellwidth=false, framevisible=false, patchsize=(24, 10), colgap=10, rowgap=0, padding=(0, 0, 0, 0))

for case in ("", "_front"), layouts in (["square"], ["square", "strip"])
    fig = Figure(size=(510, 540), fontsize=9, figure_padding=(2, 8, 2, 4))
    ylabels = ["Wall time per time step (s)", "CG iterations per time step", "Time / FFT-only time"]
    axes = Matrix{Axis}(undef, 3, length(GRIDS))

    for (col, (title, grid)) in enumerate(GRIDS)
        data = [single_benchmark_medians(grid, layout; front=case == "_front") for layout in layouts]
        _, Ny, Nz = last(data).sizes[1]
        Label(fig[-1, col], title; font=:bold, tellwidth=false)
        dimensions = length(layouts) == 1 ? "Nx² × $Nz" : "Square: Nx² × $Nz\nStrip: Nx × $Ny × $Nz"
        Label(fig[0, col], dimensions; tellwidth=false)

        axis_kw = (; titlealign=:left, titlesize=9, titlegap=2, xscale=log2, xticks=powers_of_two, xlabel="Grid points",
                     xgridvisible=false, ygridvisible=false, xminorticksvisible=false,
                     yscale=log10, yticks=decades, yminorticks=IntervalsBetween(9), yminorticksvisible=true,
                     yminortickwidth=0.5)
        axes[:, col] = [Axis(fig[row, col]; axis_kw..., title="($(letters[col + 3(row - 1)]))", ylabel=ylabels[row])
                        for row in 1:3]
        axtime, axiters, axratio = axes[:, col]

        for (d, style) in zip(data, domain_styles), (i, (name, label)) in enumerate(PRECONDITIONER_LABELS)
            kw = (; label=d === data[1] ? label : nothing, markersize, color=colors[i], marker=markers[i], style...,
                  linestyle=name == "FFT" && style.linestyle == :solid ? :dash : style.linestyle)

            scatterlines!(axtime, d.N, d.time[name]; kw...)
            name == "FFT" && continue
            scatterlines!(axiters, d.N, d.iterations[name]; kw...)
            scatterlines!(axratio, d.N, d.time[name] ./ d.time["FFT"]; kw...)
        end
    end

    for column in eachcol(axes)
        linkxaxes!(column...)
    end

    for row in 1:3
        linkyaxes!(axes[row, :]...)
    end

    for ax in axes[1:end-1, :]
        hidexdecorations!(ax; ticks=false)
    end

    for ax in axes[:, 2:end]
        hideydecorations!(ax; ticks=false, minorticks=false)
    end

    Legend(fig[4, 1:length(GRIDS)], axes[1, 1]; orientation=:horizontal, nbanks=3, legend_kw...)
    length(layouts) == 2 && Legend(fig[5, 1:length(GRIDS)], domains, ["Square domain", "Strip domain"];
                                   orientation=:horizontal, legend_kw...)

    colgap!(fig.layout, 8)
    rowgap!(fig.layout, 6)
    rowgap!(fig.layout, 1, 2)

    filename = "./Output/benchmark_single$(case)_H100" * (length(layouts) == 1 ? "" : "_square_vs_strip")
    display(fig)
    save("$filename.pdf", fig, pt_per_unit=1)
    save("$filename.png", fig, px_per_unit=6)
end
#%%
