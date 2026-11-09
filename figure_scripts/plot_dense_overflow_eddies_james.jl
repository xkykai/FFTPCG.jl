using Oceananigans
using Oceananigans.Units
using Oceananigans.ImmersedBoundaries: mask_immersed_field!
using CairoMakie

#%%
filename = "dense_overflow_Nx_1536_Ny_640_Nz_128_CG_nu4h_5.0e6_nu4z_80.0_14days"
tracer_times = [2days, 6days, 10days, 13days]
f₀ = 1e-4 # Coriolis parameter (s⁻¹)

path = "./Data/$filename/instantaneous_fields.jld2"
c_series = FieldTimeSeries(path, "c"; backend=OnDisk())
u_series, v_series = (FieldTimeSeries(path, name; backend=OnDisk(), boundary_conditions=nothing) for name in ("u", "v"))

grid = c_series.grid
Nx, Ny, Nz = size(grid)
x, y = xnodes(grid, Center()) ./ kilometers, ynodes(grid, Center()) ./ kilometers
xF, yF = xnodes(grid, Face()) ./ kilometers, ynodes(grid, Face()) ./ kilometers

bottom_tracer = map(tracer_times) do t
    c = c_series[findfirst(≈(t), c_series.times)]
    mask_immersed_field!(c, NaN)
    c = interior(c)
    [c[i, j, something(findfirst(!isnan, view(c, i, j, :)), Nz)] for i in 1:Nx, j in 1:Ny]
end

n = findfirst(≈(tracer_times[end]), c_series.times)
ζ = Field(∂x(v_series[n]) - ∂y(u_series[n]); indices=(:, :, Nz))
compute!(ζ)
mask_immersed_field!(ζ, NaN)
surface_vorticity = interior(ζ, :, :, 1) ./ f₀

#%%
let
    fig = Figure(size=(510, 330), fontsize=9, figure_padding=(4, 8, 4, 4))
    letters = 'a':'z'
    panel_title(letter, text) = rich(rich("($letter) "; font=:bold), text)
    axis_kw = (; titlefont=:regular, titlealign=:left, titlegap=2, xlabelpadding=1, ylabelpadding=1,
                 xlabel="x (km)", ylabel="y (km)", xticks=-600:200:0, yticks=-200:100:0,
                 limits=(extrema(x), extrema(y)))
    colorbar_kw = (; width=5, ticklabelsize=8, labelpadding=2, ticksize=3)
    day_label(t) = "t = $(round(Int, t / day)) days"

    axes = [Axis(fig[(q + 1) ÷ 2, 2 - q % 2]; axis_kw..., title=panel_title(letters[q], "Bottom tracer, $(day_label(t))"))
            for (q, t) in enumerate(tracer_times)]

    for (ax, c) in zip(axes, bottom_tracer)
        heatmap!(ax, x, y, c; colormap=:matter, colorrange=(0, 1), nan_color=:gray75)
    end

    Colorbar(fig[1:2, 3]; colorbar_kw..., colormap=:matter, limits=(0, 1), label="Tracer concentration")

    axζ = Axis(fig[3, 1:2]; axis_kw..., title=panel_title('e', "Surface vorticity, $(day_label(tracer_times[end]))"))
    hm = heatmap!(axζ, xF, yF, surface_vorticity; colormap=:balance, colorrange=(-1, 1), nan_color=:gray75)
    contour!(axζ, x, y, bottom_tracer[end]; levels=[0.1], color=:black, linewidth=0.5)
    Colorbar(fig[3, 3], hm; colorbar_kw..., label=rich("ζ / f", subscript("0")))

    hidexdecorations!.(axes; ticks=false)
    hideydecorations!.(axes[[2, 4]]; ticks=false)

    rowsize!(fig.layout, 3, Relative(0.5))
    rowgap!(fig.layout, 4)
    colgap!(fig.layout, 4)

    display(fig)
    save("./Output/dense_overflow_eddies_james.pdf", fig, pt_per_unit=1)
    save("./Output/dense_overflow_eddies_james.png", fig, px_per_unit=6)
end
#%%
