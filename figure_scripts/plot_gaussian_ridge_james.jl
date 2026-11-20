using Oceananigans
using Oceananigans.Units
using Oceananigans.ImmersedBoundaries: mask_immersed_field!
using CairoMakie
using NaNStatistics

#%%
T₀, S₀, c₀ = 20, 35, 1
buoyancy = SeawaterBuoyancy()
α = buoyancy.equation_of_state.thermal_expansion
β = buoyancy.equation_of_state.haline_contraction
b₀ = buoyancy.gravitational_acceleration * (α * T₀ - β * S₀)

data_path(solver, kind) = "./Data/gaussian_ridge_$(solver)_mwe_fields_constantS_noslip_$(kind).jld2"

function load_solver_output(solver)
    T, S, b, c, p, d = (FieldTimeSeries(data_path(solver, "fields"), name) for name in ("T", "S", "b", "c", "p", "d"))
    Tbar, Sbar, cbar = (FieldTimeSeries(data_path(solver, "averaged"), name) for name in ("Tbar", "Sbar", "cbar"))

    for field in (T, S, b, c, p, d), n in eachindex(field.times)
        mask_immersed_field!(field[n], NaN)
    end

    slice(field, n=Nt) = interior(field[n], :, 1, :)
    Nt = length(T.times)
    normalized(φ) = φ ./ φ[1]
    averaged(field) = interior(field, 1, 1, 1, :)

    anomalies = (T = slice(T) .- T₀,
                 S = slice(S) .- S₀,
                 b = slice(b) .- b₀,
                 c = slice(c) .- c₀,
                 p = slice(p) .- slice(p, 1),
                 d = slice(d))

    bbar = [nanmean(interior(b[n])) for n in 1:Nt]

    normalized_averages = (T = normalized(averaged(Tbar)),
                           S = normalized(averaged(Sbar)),
                           b = normalized(bbar),
                           c = normalized(averaged(cbar)))

    return (; T.grid, T.times, anomalies, normalized_averages)
end

cg = load_solver_output("cg")
fft = load_solver_output("fft")

xC = xnodes(cg.grid, Center())
zC = znodes(cg.grid, Center())

#%%
superscript_minus(e) = replace(string(e), "-" => "−")

function scale_to_unit_order(φ)
    e = floor(Int, log10(nanmaximum(abs.(φ))))
    e = e < -1 ? e : 0
    return φ ./ 10.0^e, e
end

panels = [(:T, ["Temperature anomaly"],    "°C",     :balance),
          (:S, ["Salinity anomaly"],       "psu",    Reverse(:balance)),
          (:b, ["Buoyancy anomaly"],       "m s⁻²",  :balance),
          (:c, ["Passive tracer anomaly"], "",       :curl),
          (:p, ["Pressure deviation"],     "m² s⁻²", :diff),
          (:d, ["Divergence"],             "s⁻¹",    :balance)]

solvers = [(cg, "FFT-preconditioned CG"), (fft, "FFT only")]

let
    fig = Figure(size=(510, 900), fontsize=9, figure_padding=(4, 8, 4, 4))

    for (j, (_, title)) in enumerate(solvers)
        Label(fig[0, 2j-1:2j], title; font=:bold, fontsize=12, tellwidth=false)
    end

    axes = Matrix{Axis}(undef, length(panels), length(solvers))
    letters = 'a':'z'

    for (i, (name, symbol, units, colormap)) in enumerate(panels), (j, (output, _)) in enumerate(solvers)

        φ, e = scale_to_unit_order(output.anomalies[name])
        φmax = nanmaximum(abs.(φ))

        title = rich(rich("($(letters[2(i-1) + j])) "; font=:bold), symbol...)
        label = e == 0 ? units : rich("×10", superscript(superscript_minus(e)), " ", units)

        ax = Axis(fig[i, 2j-1];
                  title,
                  titlefont = :regular,
                  titlealign = :left,
                  titlesize = 11,
                  titlegap = 2,
                  aspect = 2,
                  xticks = -60:30:60,
                  yticks = -30:10:0,
                  xlabel = "x (m)",
                  ylabel = "z (m)",
                  xlabelpadding = 1,
                  ylabelpadding = 1)

        hm = heatmap!(ax, xC, zC, φ; colormap, colorrange=(-φmax, φmax), nan_color=:gray75)
        Colorbar(fig[i, 2j], hm; label, width=6, ticklabelsize=8, labelpadding=2, ticksize=3)

        axes[i, j] = ax
    end

    linkaxes!(axes...)

    for ax in axes[1:end-1, :]
        hidexdecorations!(ax, ticks=false, grid=false)
    end

    for ax in axes[:, 2]
        hideydecorations!(ax, ticks=false, grid=false)
    end

    variable_colors = Makie.wong_colors()[[1, 2, 3, 6]]
    variable_linestyles = [:solid, :dash, :dot, :dashdot]
    time_axes = Axis[]
    offset_labels = [n == 0 ? rich("1") : rich("1 $(n < 0 ? "−" : "+") $(abs(n) == 1 ? "" : "$(abs(n))×")10", superscript("−6"))
                     for n in -2:2]

    for (j, (output, _)) in enumerate(solvers)
        ax = Axis(fig[length(panels) + 1, 2j-1:2j];
                  title = "($(letters[2length(panels) + j]))",
                  titlealign = :left,
                  titlesize = 11,
                  titlegap = 2,
                  xlabel = "Time (hours)",
                  ylabel = "⟨φ⟩(t) / ⟨φ⟩(0)",
                  yticks = (1 .+ (-2:2) .* 1e-6, offset_labels),
                  limits = (0, output.times[end] / hours, 1 - 2.5e-6, 1 + 2.5e-6),
                  xgridvisible = false,
                  ygridvisible = false)

        for (k, name) in enumerate((:T, :S, :b, :c))
            lines!(ax, output.times ./ hours, output.normalized_averages[name];
                   color = variable_colors[k], linewidth = 2.5,
                   linestyle = variable_linestyles[k],
                   label = string(name))
        end

        push!(time_axes, ax)
    end

    hideydecorations!(time_axes[2], ticks=false, grid=false)
    axislegend(time_axes[1]; orientation=:horizontal, position=:rb, framevisible=false, patchsize=(16, 8))

    rowsize!(fig.layout, length(panels) + 1, Relative(0.12))
    rowgap!(fig.layout, 6)
    colgap!(fig.layout, 4)
    colgap!(fig.layout, 2, 14)

    display(fig)

    save("./Output/gaussian_ridge_james.pdf", fig, pt_per_unit=1)
    save("./Output/gaussian_ridge_james.png", fig, px_per_unit=6)
end
#%%
