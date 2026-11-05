using Oceananigans
using Oceananigans.ImmersedBoundaries: mask_immersed_field!
using CairoMakie
using NaNStatistics

#%%
filename = "rough_RB_seaiceformation_noslip_periodic_halfdomain_stretched_Ra_1.0e8_Pr_1_Nr_8_Lx_1_Lz_1_Nx_1024_Nz_512"
snapshot_time = 1000

function load_solver_output(solver)
    data_dir = "./Data/$(filename)_$(solver)"
    fields = NamedTuple(Symbol(name) => FieldTimeSeries("$data_dir/instantaneous_fields.jld2", name; backend=OnDisk())
                        for name in ("b", "p", "d"))
    Nu = FieldTimeSeries("$data_dir/averaged_fields.jld2", "Nu")
    KE = FieldTimeSeries("$data_dir/KE_fields.jld2", "KE")

    n = findfirst(==(snapshot_time), fields.b.times)
    b, p, d = map(fields) do field
        φ = field[n]
        mask_immersed_field!(φ, NaN)
        interior(φ, :, 1, :)
    end

    return (; b, p = p .- nanmean(p), d,
              Nu = interior(Nu[end], 1, 1, :), Nu.grid, KE_times = KE.times, KE = interior(KE, 1, 1, 1, :))
end

cg = load_solver_output("CG")
fft = load_solver_output("FFT")

xC = xnodes(cg.grid, Center())
zF = znodes(cg.grid, Face())

#%%
superscript_minus(e) = replace(string(e), "-" => "−")

pmax = nanmaximum(abs.([cg.p; fft.p]))

panels = [(:b, "Buoyancy",                 Reverse(:roma), nanextrema([cg.b; fft.b]), ((0, 1), (0, 1))),
          (:p, "Pressure deviation",       :vik,           (-pmax, pmax),               ((0, 1), (0, 1))),
          (:d, "Divergence in box",        :balance,       nothing,                     ((0.15, 0.35), (0.45, 1)))]

solvers = [(cg, "FFT-preconditioned CG", Makie.wong_colors()[3]),
           (fft, "FFT only (inexact solution)", :gray60)]

let
    fig = Figure(size=(510, 520), fontsize=9, figure_padding=(4, 8, 4, 4))
    letters = 'a':'z'
    axes = Matrix{Axis}(undef, length(solvers), length(panels))

    for (i, (output, title, _)) in enumerate(solvers)
        Label(fig[2i-1, 1:2length(panels)], title; font=:bold, fontsize=10, tellwidth=false)

        for (j, (name, quantity, colormap, colorrange, limits)) in enumerate(panels)
            ax = Axis(fig[2i, 2j-1];
                      title = rich(rich("($(letters[length(panels)*(i-1) + j])) "; font=:bold), quantity),
                      titlefont = :regular,
                      titlealign = :left,
                      titlegap = 2,
                      aspect = 1,
                      limits,
                      xticks = WilkinsonTicks(3, k_min=2, k_max=3),
                      yticks = 0:0.25:1,
                      xlabel = "x",
                      ylabel = "z")

            φ = output[name]
            colorrange = something(colorrange, (-1, 1) .* nanpctile(abs.(φ), 99))
            e = floor(Int, log10(last(colorrange)))
            e = e < -1 ? e : 0
            φ, colorrange = φ ./ 10.0^e, colorrange ./ 10.0^e
            label = e == 0 ? "" : rich("×10", superscript(superscript_minus(e)))

            hm = heatmap!(ax, xC, zF, φ; colormap, colorrange, nan_color=:gray75)
            Colorbar(fig[2i, 2j], hm; label, width=5, ticklabelsize=8, labelpadding=2, ticksize=3)
            axes[i, j] = ax
        end
    end

    linkaxes!(axes[:, 1:2]...)
    linkaxes!(axes[:, 3]...)

    (x₁, x₂), (z₁, z₂) = last(last(panels))
    for ax in axes[:, 2]
        lines!(ax, [x₁, x₂, x₂, x₁, x₁], [z₁, z₁, z₂, z₂, z₁]; color=:black, linewidth=0.8)
    end

    for ax in axes[1, :]
        hidexdecorations!(ax, ticks=false)
    end

    for ax in axes[:, 2]
        hideydecorations!(ax, ticks=false)
    end

    averages = fig[2length(solvers) + 1, 1:2length(panels)] = GridLayout()
    axis_kw = (; titlefont=:regular, titlealign=:left, titlegap=2, xgridvisible=false, ygridvisible=false)
    axNu = Axis(averages[1, 1]; axis_kw..., title=rich(rich("(g) "; font=:bold), "Nusselt number, 500 ≤ t ≤ 1000"),
                xlabel="Nu", ylabel="z", yticks=0:0.5:1, limits=(nothing, (0, 1)))
    axKE = Axis(averages[1, 2]; axis_kw..., title=rich(rich("(h) "; font=:bold), "Kinetic energy"),
                xlabel="t", ylabel="⟨KE⟩")

    handles = map(reverse(solvers)) do (output, _, color)
        lines!(axNu, output.Nu, zF; color, linewidth=1.8)
        lines!(axKE, output.KE_times, output.KE; color, linewidth=1.8)
    end

    Legend(fig[2length(solvers) + 2, 1:2length(panels)], reverse(handles), [title for (_, title, _) in solvers];
           orientation=:horizontal, tellwidth=false, framevisible=false, patchsize=(24, 10), colgap=10,
           padding=(0, 0, 0, 0))

    rowsize!(fig.layout, 2length(solvers) + 1, Relative(0.28))
    rowgap!(fig.layout, 4)
    colgap!(fig.layout, 4)

    display(fig)
    save("./Output/rough_rayleigh_benard_james.pdf", fig, pt_per_unit=1)
    save("./Output/rough_rayleigh_benard_james.png", fig, px_per_unit=6)
end
#%%
