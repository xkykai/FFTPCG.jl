using Oceananigans
using Oceananigans.Units
using Oceananigans.ImmersedBoundaries: mask_immersed_field!
using CairoMakie
using LinearAlgebra: cross, normalize

#%%
filename = "dense_overflow_Nx_1536_Ny_640_Nz_128_CG_nu4h_5.0e6_nu4z_80.0_14days"
snapshot_time = 14days
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
y_slice = (-100, 10)

bottom_height = grid.immersed_boundary.bottom_height[1:Nx, 1:Ny, 1] ./ kilometers

hₑ = 600    # embayment depth (m)
tanα = 0.01 # bottom slope
h₀ = 300    # inflow thickness at the embayment's western wall (m)
Lρ = sqrt(0.019 * h₀) / 1e-4 / kilometers

# Inflow at the northern boundary (Legg et al., 2006), with xʷ the distance from the embayment's western wall
function F(zstar)
    Riₘ = 1 / 3
    zstar >= Riₘ / (2 - Riₘ) && return 1.0
    zstar <= -Riₘ / (2 + Riₘ) && return 0.0
    return zstar / (Riₘ * (zstar + 1)) + 1 / 2
end

inflow_x = range(-50, 50, length=201)
inflow_z = range(-hₑ, 0, length=241)
inflow_h = h₀ .* exp.(-(inflow_x .+ 50) ./ Lρ)
inflow_v = [exp(-(xᵢ + 50) / Lρ) * (1 - F((zₖ + hₑ - h) / h)) for (xᵢ, h) in zip(inflow_x, inflow_h), zₖ in inflow_z]

# Arrow from `from` to `to`, with heads sized in the axis box and turned to face the camera
function arrow3d!(ax, from, to; both_ends=false, headlength=0.02, color=:black, linewidth=0.8)
    from, to = Point3f(from), Point3f(to)
    scale = Vec3f(ax.aspect[] ./ map(limit -> limit[2] - limit[1], ax.limits[]))
    camera = Vec3f(cos(ax.elevation[]) * cos(ax.azimuth[]), cos(ax.elevation[]) * sin(ax.azimuth[]), sin(ax.elevation[]))
    lines!(ax, [from, to]; color, linewidth)

    for (tail, tip) in (both_ends ? ((from, to), (to, from)) : ((from, to),))
        direction = normalize((tip - tail) .* scale)
        side = 0.35headlength * normalize(cross(direction, camera))
        base = tip .* scale - headlength * direction
        mesh!(ax, Point3f[tip, (base + side) ./ scale, (base - side) ./ scale], [1 2 3]; color, shading=NoShading)
    end
end

#%%
let
    fig = Figure(size=(510, 300), fontsize=9, figure_padding=(4, 8, 4, 4))
    panel_title(letter, text) = rich(rich("($letter) "; font=:bold), text)
    axis_kw = (; titlefont=:regular, titlealign=:left, titlegap=2, xlabelpadding=1, ylabelpadding=1)
    tracer_kw = (; colormap=:matter, colorrange=(0, 1), nan_color=:gray75)
    label_kw = (; fontsize=7, color=:gray20)
    colorbar_kw = (; width=5, ticklabelsize=8, labelpadding=2, ticksize=3)
    days = round(Int, snapshot_time / day)
    x₀, x₁ = x[1] - 0.25, x[end] + 0.25
    y₀, y₁ = y[1] - 0.25, y[end] + 0.25
    H = -bottom_height[1, 1]

    setup_x, setup_y = (x₀ - 100, x₁ + 40), (y₀ - 40, y₁)

    axsetup = Axis3(fig[1, 1];
                    title = panel_title('a', "Setup"),
                    titlefont = :regular,
                    titlealign = :left,
                    titlegap = 2,
                    aspect = (1, (setup_y[2] - setup_y[1]) / (setup_x[2] - setup_x[1]), 0.3),
                    limits = (setup_x, setup_y, (-3.4, 0.15)),
                    azimuth = -0.6π,
                    elevation = 0.08π,
                    perspectiveness = 0,
                    viewmode = :fitzoom,
                    protrusions = (14, 14, 25, 12))
    hidedecorations!(axsetup)
    hidespines!(axsetup)

    surface!(axsetup, x, y, bottom_height; color=bottom_height, colormap=[:gray55, :gray92], colorrange=(-3.4, 0), rasterize=5)
    surface!(axsetup, fill(x₀, Ny, 2), [y y], [fill(-H, Ny) bottom_height[1, :]];
             color=zeros(Ny, 2), colormap=[:gray45, :gray45], rasterize=5)

    # Slope angle α on the western face, drawn in the axis box so it matches the exaggerated slope
    box_scale = axsetup.aspect[] ./ map(limit -> limit[2] - limit[1], axsetup.limits[])
    α_drawn = atan(tanα * box_scale[3] / box_scale[2])
    arc_radius = 0.07
    θ = range(0, α_drawn, length=20)
    lines!(axsetup, fill(x₀ - 1, 20), y₀ .+ arc_radius .* cos.(θ) ./ box_scale[2], -H .+ arc_radius .* sin.(θ) ./ box_scale[3];
           color=:white, linewidth=0.6)
    text!(axsetup, Point3f(x₀ - 1, y₀ + 1.6arc_radius * cos(α_drawn / 2) / box_scale[2], -H + 1.6arc_radius * sin(α_drawn / 2) / box_scale[3]);
          text="α", color=:white, fontsize=7, align=(:center, :center))

    outline = Point3f[(x₀, y₀, -H), (x₀, y₀, 0), (x₁, y₀, 0), (x₁, y₀, -H), (NaN, NaN, NaN),
                      (x₀, y₀, 0), (x₀, y₁, 0), (x₁, y₁, 0), (x₁, y₀, 0)]
    lines!(axsetup, outline; color=:gray60, linewidth=0.4)

    arrow3d!(axsetup, (0, y₁, -0.3), (0, -25, -0.6))
    arrow3d!(axsetup, (-580, -150, -1.6), (x₀ - 90, -150, -1.6))
    text!(axsetup, Point3f(0, -25, -0.6); text="Inflow", align=(:left, :top), offset=(3, 0), label_kw...)
    text!(axsetup, Point3f(-600, -150, -1.4); text="Open boundary", align=(:left, :bottom), label_kw...)
    text!(axsetup, Point3f(x₀, y₀, -H); text="tan α = $tanα", align=(:right, :top), offset=(10, -3), label_kw...)

    dimension_kw = (; both_ends=true, headlength=0.012, color=:gray20, linewidth=0.5)
    arrow3d!(axsetup, (x₀, y₀ - 12, -H), (x₁, y₀ - 12, -H); dimension_kw...)
    arrow3d!(axsetup, (x₀ - 12, y₀, -H), (x₀ - 12, y₁, -H); dimension_kw...)
    arrow3d!(axsetup, (x₁ + 12, y₀, -H), (x₁ + 12, y₀, 0); dimension_kw...)
    arrow3d!(axsetup, (-50, y₁ - 5, 0.08), (50, y₁ - 5, 0.08); dimension_kw...)
    arrow3d!(axsetup, (x₁, 0, 0.1), (x₁, y₁, 0.1); dimension_kw...)
    arrow3d!(axsetup, (-50, 0, -hₑ / kilometers), (-50, 0, 0); dimension_kw...)
    text!(axsetup, Point3f((x₀ + x₁) / 2, y₀ - 12, -H); text="$(round(Int, x₁ - x₀)) km", align=(:center, :top), offset=(0, -1), label_kw...)
    text!(axsetup, Point3f(x₀ - 12, (y₀ + y₁) / 2, -H); text="$(round(Int, y₁ - y₀)) km", align=(:right, :center), offset=(-3, 0), label_kw...)
    text!(axsetup, Point3f(x₁ + 12, y₀, -H / 2); text="$(round(Int, H * kilometers)) m", align=(:left, :center), offset=(3, 0), label_kw...)
    text!(axsetup, Point3f(0, y₁, 0.1); text="100 km", align=(:center, :bottom), offset=(0, 2), label_kw...)
    text!(axsetup, Point3f(x₁, y₁ / 2, 0.1); text="$(round(Int, y₁)) km", align=(:left, :bottom), offset=(5, 1), label_kw...)
    text!(axsetup, Point3f(-50, 0, -hₑ / 2kilometers); text="$hₑ m", align=(:right, :center), offset=(-6, 0), label_kw...)

    axinflow = Axis(fig[1, 2]; axis_kw...,
                    title = panel_title('b', "Inflow at y = $(round(Int, y₁)) km"),
                    xlabel = "x (km)",
                    ylabel = "z (m)",
                    xticks = -50:50:50,
                    yticks = -600:200:0,
                    limits = ((-50, 50), (-hₑ, 0)))

    hmv = heatmap!(axinflow, inflow_x, inflow_z, inflow_v; colormap=:tempo, colorrange=(0, 1))
    Colorbar(fig[1, 3], hmv; label="−v / U₀", colorbar_kw...)

    axmap = Axis(fig[2, 1]; axis_kw...,
                 title = panel_title('c', "Tracer in the bottom cell, t = $days days"),
                 xlabel = "x (km)",
                 ylabel = "y (km)",
                 xticks = -600:200:0,
                 yticks = -200:100:0,
                 limits = (extrema(x), extrema(y)))

    hm = heatmap!(axmap, x, y, bottom(c); tracer_kw..., rasterize=5)
    contour!(axmap, x, y, bottom(b); levels=-0.018:0.0015:0, color=:black, linewidth=0.3)
    lines!(axmap, [x[i_slice], x[i_slice]], [y_slice...]; color=:black, linewidth=0.8, linestyle=:dash)

    axslice = Axis(fig[2, 2]; axis_kw...,
                   title = panel_title('d', "Tracer at x = $(replace(string(round(Int, x[i_slice])), "-" => "−")) km"),
                   xlabel = "y (km)",
                   ylabel = "z (m)",
                   xticks = -100:50:0,
                   yticks = -1500:500:0,
                   limits = (y_slice, (-1700, 0)))

    heatmap!(axslice, y, z, c[i_slice, :, :]; tracer_kw...)
    Colorbar(fig[2, 3], hm; label="Tracer concentration", colorbar_kw...)

    colsize!(fig.layout, 1, Aspect(2, grid.Lx / grid.Ly))
    colgap!(fig.layout, 1, 12)
    colgap!(fig.layout, 2, 4)
    rowgap!(fig.layout, 6)

    display(fig)
    save("./Output/dense_overflow_james.pdf", fig, pt_per_unit=1)
    save("./Output/dense_overflow_james.png", fig, px_per_unit=6)
end
#%%
