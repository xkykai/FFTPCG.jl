using Oceananigans
using Oceananigans.Units
using Oceananigans.ImmersedBoundaries: mask_immersed_field!
using CairoMakie
using Statistics

#%%
filename = "dense_overflow_Nx_1536_Ny_640_Nz_128_CG_nu4h_5.0e6_nu4z_80.0_14days"
averaging_start = 7days
plume_threshold = 0.01

Δb₀ = 0.019   # inflow buoyancy anomaly (m s⁻²)
h₀ = 300      # inflow thickness (m)
f₀ = 1e-4     # Coriolis parameter (s⁻¹)
N² = 2.3e-3^2 # ambient stratification (s⁻²)

path = "./Data/$filename/instantaneous_fields.jld2"
c_series, b_series = (FieldTimeSeries(path, name; backend=OnDisk()) for name in ("c", "b"))
u_series = FieldTimeSeries(path, "u"; backend=OnDisk(), boundary_conditions=nothing)

grid = c_series.grid
Nx, Ny, Nz = size(grid)
Δx, Δy, Δz = grid.Lx / Nx, grid.Ly / Ny, grid.Lz / Nz
x = xnodes(grid, Center()) ./ kilometers

b_classes = range(-0.021, 0.006, step=5e-4)
c_classes = range(-0.4, 1.2, step=0.02)

function accumulate_census!(tracer_area, volume, c, b, b_classes, c_classes)
    for I in CartesianIndices(c)
        isnan(c[I]) && continue
        b_class = searchsortedlast(b_classes, b[I])
        c_class = searchsortedlast(c_classes, c[I])
        tracer_area[I[1], b_class] += c[I]
        volume[c_class, b_class] += 1
    end
end

snapshots = findall(≥(averaging_start), c_series.times)
transport = zeros(Nx, length(snapshots))
tracer_area = zeros(Nx, length(b_classes) - 1)
volume = zeros(length(c_classes) - 1, length(b_classes) - 1)

for (m, n) in enumerate(snapshots)
    c, b = map((c_series, b_series)) do series
        φ = series[n]
        mask_immersed_field!(φ, NaN)
        interior(φ)
    end

    u = interior(u_series[n])
    plume = c .> plume_threshold
    transport[:, m] .= @views vec(sum((u[1:Nx, :, :] .+ u[2:Nx+1, :, :]) .* plume, dims=(2, 3))) .* (Δy * Δz / 2)
    accumulate_census!(tracer_area, volume, c, b, b_classes, c_classes)
end

tracer_area .*= Δy * Δz / length(snapshots)
volume .*= Δx * Δy * Δz / length(snapshots)

#%%
let
    fig = Figure(size=(510, 330), fontsize=9, figure_padding=(4, 8, 4, 4))
    panel_title(letter, text) = rich(rich("($letter) "; font=:bold), text)
    axis_kw = (; titlefont=:regular, titlealign=:left, titlegap=2, xlabelpadding=1, ylabelpadding=1,
                 xgridvisible=false, ygridvisible=false)
    colorbar_kw = (; width=5, ticklabelsize=8, labelpadding=2, ticksize=3)
    centers(classes) = (classes[1:end-1] .+ classes[2:end]) ./ 2
    b_ticks = -20:5:0
    b_label = rich("Buoyancy (10", superscript("−3"), " m s", superscript("−2"), ")")
    period = "days $(round(Int, averaging_start / day))–$(round(Int, c_series.times[end] / day))"

    axT = Axis(fig[1, 1]; axis_kw...,
               title = panel_title('a', "Along-slope transport of fluid with tracer > $plume_threshold"),
               ylabel = "Transport (Sv)")

    for m in axes(transport, 2)
        lines!(axT, x, transport[:, m] ./ 1e6; color=(:gray50, 0.5), linewidth=0.5)
    end

    lines!(axT, x, vec(mean(transport, dims=2)) ./ 1e6; color=:black, linewidth=1.8, label="Mean, $period")
    hlines!(axT, -Δb₀ * h₀^2 / 2f₀ / 1e6; color=Makie.wong_colors()[6], linewidth=1, linestyle=:dash, label="Inflow")
    axislegend(axT; position=:lb, framevisible=false, patchsize=(16, 8), rowgap=0, padding=(4, 0, 0, 0))

    axxb = Axis(fig[2, 1]; axis_kw...,
                title = panel_title('c', "Tracer by buoyancy class, $period"),
                xlabel = "x (km)",
                ylabel = b_label,
                xticks = -600:200:0,
                yticks = b_ticks)

    hmxb = heatmap!(axxb, x, centers(b_classes) .* 1e3, tracer_area ./ kilometers^2;
                    colormap=:dense, colorrange=(0, 2))
    Colorbar(fig[2, 2], hmxb; colorbar_kw..., label=rich("Tracer area per class (km", superscript("2"), ")"))
    hlines!(axxb, [-10, -9]; color=:black, linewidth=0.6, linestyle=:dot)
    text!(axxb, x[1] + 10, -9; text="Legg et al. (2006)", align=(:left, :bottom), fontsize=7)

    plume = centers(c_classes) .> plume_threshold
    axV = Axis(fig[1, 3]; axis_kw...,
               title = panel_title('b', "Plume volume, $period"),
               ylabel = rich("Volume (10", superscript("3"), " km", superscript("3"), ")"),
               limits = ((0, 1), (0, nothing)))

    stairs!(axV, centers(c_classes)[plume], vec(sum(volume, dims=2))[plume] ./ kilometers^3 ./ 1e3;
            step=:center, color=:black, linewidth=1)

    axcb = Axis(fig[2, 3]; axis_kw...,
                title = panel_title('d', "Volume by class"),
                xlabel = "Tracer concentration",
                yticks = b_ticks,
                limits = ((0, 1), nothing))

    hmcb = heatmap!(axcb, centers(c_classes), centers(b_classes) .* 1e3, replace(volume, 0 => NaN) ./ kilometers^3;
                    colormap=:dense, colorscale=log10, colorrange=(1, 1e3))
    Colorbar(fig[2, 4], hmcb; colorbar_kw..., label=rich("Volume per class (km", superscript("3"), ")"))

    for ambient_depth in 600:600:3000
        ambient_buoyancy = -N² * ambient_depth * 1e3
        lines!(axcb, [0, 1], [ambient_buoyancy, -Δb₀ * 1e3]; color=:black, linewidth=0.5, linestyle=:dash)
        text!(axcb, 0.03, ambient_buoyancy; text="$ambient_depth m", align=(:left, :bottom), fontsize=7)
    end

    linkxaxes!(axT, axxb)
    linkxaxes!(axV, axcb)
    linkyaxes!(axxb, axcb)
    ylims!(axxb, -21, 0)
    hidexdecorations!(axT, ticks=false)
    hidexdecorations!(axV, ticks=false)
    hideydecorations!(axcb, ticks=false)

    rowsize!(fig.layout, 1, Relative(0.3))
    colsize!(fig.layout, 3, Relative(0.32))
    rowgap!(fig.layout, 6)
    colgap!(fig.layout, 4)
    colgap!(fig.layout, 2, 12)

    display(fig)
    save("./Output/dense_overflow_mixing_james.pdf", fig, pt_per_unit=1)
    save("./Output/dense_overflow_mixing_james.png", fig, px_per_unit=6)
end
#%%
