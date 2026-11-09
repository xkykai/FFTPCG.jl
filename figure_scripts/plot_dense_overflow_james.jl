using Oceananigans
using Oceananigans.Units
using Oceananigans.ImmersedBoundaries: mask_immersed_field!
using CairoMakie

#%%
filename = "dense_overflow_Nx_1536_Ny_640_Nz_128_CG_nu4h_5.0e6_nu4z_80.0_14days"
snapshot_time = 13days
x_slice = -50kilometers

fields = NamedTuple(Symbol(name) => FieldTimeSeries("./Data/$filename/instantaneous_fields.jld2", name; backend=OnDisk())
                    for name in ("c", "b"))

n = findfirst(≈(snapshot_time), fields.c.times)
c, b = map(fields) do field
    φ = field[n]
    mask_immersed_field!(φ, NaN)
    interior(φ)
end

grid = fields.c.grid
Nx, Ny, Nz = size(grid)
x, y, z = nodes(grid, Center(), Center(), Center())
x, y = x ./ kilometers, y ./ kilometers

bottom_k = [something(findfirst(!isnan, view(c, i, j, :)), Nz) for i in 1:Nx, j in 1:Ny]
bottom(φ) = [φ[i, j, bottom_k[i, j]] for i in 1:Nx, j in 1:Ny]
i_slice = searchsortedlast(x, x_slice / kilometers)

#%%
let
    fig = Figure(size=(510, 158), fontsize=9, figure_padding=(4, 8, 4, 4))
    panel_title(letter, text) = rich(rich("($letter) "; font=:bold), text)
    axis_kw = (; titlefont=:regular, titlealign=:left, titlegap=2, xlabelpadding=1, ylabelpadding=1)
    tracer_kw = (; colormap=:matter, colorrange=(0, 1), nan_color=:gray75)
    days = round(Int, snapshot_time / day)

    axmap = Axis(fig[1, 1]; axis_kw...,
                 title = panel_title('a', "Tracer in the bottom cell, t = $days days"),
                 xlabel = "x (km)",
                 ylabel = "y (km)",
                 xticks = -600:200:0,
                 yticks = -200:100:0,
                 limits = (extrema(x), extrema(y)))

    hm = heatmap!(axmap, x, y, bottom(c); tracer_kw...)
    contour!(axmap, x, y, bottom(b); levels=-0.018:0.0015:0, color=:black, linewidth=0.3)
    lines!(axmap, [x[i_slice], x[i_slice]], [-100, 0]; color=:black, linewidth=0.8, linestyle=:dash)

    axslice = Axis(fig[1, 2]; axis_kw...,
                   title = panel_title('b', "Tracer at x = −$(round(Int, abs(x_slice) / kilometers)) km"),
                   xlabel = "y (km)",
                   ylabel = "z (m)",
                   xticks = -100:50:0,
                   yticks = -1500:500:0,
                   limits = (-100, 10, -1700, 0))

    heatmap!(axslice, y, z, c[i_slice, :, :]; tracer_kw...)
    Colorbar(fig[1, 3], hm; label="Tracer concentration", width=5, ticklabelsize=8, labelpadding=2, ticksize=3)

    colsize!(fig.layout, 1, Aspect(1, grid.Lx / grid.Ly))
    colgap!(fig.layout, 1, 12)
    colgap!(fig.layout, 2, 4)

    display(fig)
    save("./Output/dense_overflow_james.pdf", fig, pt_per_unit=1)
    save("./Output/dense_overflow_james.png", fig, px_per_unit=6)
end
#%%
