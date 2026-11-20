using Oceananigans
using Oceananigans.Units
using Oceananigans.ImmersedBoundaries: mask_immersed_field!
using CairoMakie

#%%
filename = "dense_overflow_Nx_1536_Ny_640_Nz_128_CG_nu4h_5.0e6_nu4z_80.0_14days"
snapshot_times = [2days, 6days, 10days, 14days]
plume_threshold = 0.05
buoyancy_levels = -0.018:0.0015:0
hₑ = 600    # embayment depth (m)
tanα = 0.01 # bottom slope

fields = NamedTuple(Symbol(name) => FieldTimeSeries("./Data/$filename/instantaneous_fields.jld2", name; backend=OnDisk())
                    for name in ("c", "b"))

grid = fields.c.grid
Nx, Ny, Nz = size(grid)
x, y, z = nodes(grid, Center(), Center(), Center()) ./ kilometers
Δz = grid.Lz / Nz / kilometers

wet = grid.immersed_boundary.bottom_height[1:Nx, 1:Ny, 1] .< 0
bottom_height = [wet[i, j] ? -hₑ / kilometers + tanα * min(y[j], 0) : 0.0 for i in 1:Nx, j in 1:Ny]

# Height (km) and buoyancy of the surface where the tracer falls below `plume_threshold`,
# and the bottom buoyancy of the slope outside the plume
function plume_surface(n)
    c, b = map(fields) do field
        φ = field[n]
        mask_immersed_field!(φ, NaN)
        interior(φ)
    end

    height, buoyancy, slope_buoyancy = (fill(NaN, Nx, Ny) for _ in 1:3)

    for j in 1:Ny, i in 1:Nx
        wet[i, j] || continue
        k = findlast(>(plume_threshold), view(c, i, j, :))

        if isnothing(k)
            slope_buoyancy[i, j] = b[i, j, findfirst(!isnan, view(b, i, j, :))]
        else
            above = k < Nz ? (c[i, j, k] - plume_threshold) / (c[i, j, k] - c[i, j, k+1]) : 0
            height[i, j] = max(z[k] + above * Δz, bottom_height[i, j])
            buoyancy[i, j] = b[i, j, k] + above * (b[i, j, min(k + 1, Nz)] - b[i, j, k])
        end
    end

    return (; height, buoyancy, slope_buoyancy)
end

# Contours of `buoyancy` as lines lying on the surface `height`, broken where `height` steps by more than 50 m
function buoyancy_contours(buoyancy, height)
    points = Point3f[]
    contours = Makie.Contours.contours(x, y, buoyancy, buoyancy_levels)

    for level in Makie.Contours.levels(contours), line in Makie.Contours.lines(level)
        for (x_point, y_point) in zip(Makie.Contours.coordinates(line)...)
            i, j = clamp(searchsortedlast(x, x_point), 1, Nx), clamp(searchsortedlast(y, y_point), 1, Ny)
            height_step = max(abs(height[min(i + 1, Nx), j] - height[i, j]), abs(height[i, min(j + 1, Ny)] - height[i, j]))
            push!(points, height_step > 0.05 ? Point3f(NaN) : Point3f(x_point, y_point, height[i, j] + 0.02))
        end

        push!(points, Point3f(NaN))
    end

    return points
end

#%%
let
    fig = Figure(size=(510, 250), fontsize=9, figure_padding=(4, 8, 4, 4))
    letters = 'a':'z'
    buoyancy_kw = (; colormap=:haline, colorrange=(-14, 0))

    axis_style = (; labelsize=8, labelcolor=:gray30, ticklabelsize=7, ticklabelcolor=:gray30, ticksize=2, tickwidth=0.4,
                    tickcolor=:gray50, spinewidth=0.4, spinecolor_1=:gray50, spinecolor_2=:transparent,
                    spinecolor_3=:transparent, gridvisible=false)
    axis_kw = (; (Symbol(dimension, name) => value for dimension in "xyz" for (name, value) in pairs(axis_style))...)

    for (q, t) in enumerate(snapshot_times)
        plume = plume_surface(findfirst(≈(t), fields.c.times))

        ax = Axis3(fig[(q + 1) ÷ 2, 2 - q % 2]; axis_kw...,
                   title = rich(rich("($(letters[q])) "; font=:bold), "t = $(round(Int, t / day)) days"),
                   titlefont = :regular,
                   titlealign = :left,
                   titlegap = 2,
                   xlabel = "x (km)",
                   ylabel = "y (km)",
                   zlabel = "z (km)",
                   xticks = -500:250:0,
                   yticks = -200:100:0,
                   zticks = -3:1:0,
                   aspect = (1, grid.Ly / grid.Lx, 0.18),
                   limits = (extrema(x), extrema(y), (-3.3, 0)),
                   azimuth = -0.6π,
                   elevation = 0.14π,
                   perspectiveness = 0.3,
                   protrusions = (30, 4, 16, 10),
                   xlabeloffset = 20,
                   ylabeloffset = 30,
                   zlabeloffset = 24)

        surface!(ax, x, y, bottom_height; color=zeros(Nx, Ny), colormap=[:gray78, :gray78], rasterize=5)
        surface!(ax, x, y, plume.height; color=plume.buoyancy .* 1e3, buoyancy_kw..., rasterize=5)
        lines!(ax, buoyancy_contours(plume.buoyancy, plume.height); color=:black, linewidth=0.4)
        lines!(ax, buoyancy_contours(plume.slope_buoyancy, bottom_height); color=:black, linewidth=0.4)
    end

    Colorbar(fig[1:2, 3]; buoyancy_kw..., width=5, ticklabelsize=8, labelpadding=2, ticksize=3, height=Relative(0.6),
             label=rich("Buoyancy (10", superscript("−3"), " m s", superscript("−2"), ")"))

    rowgap!(fig.layout, 2)
    colgap!(fig.layout, 4)

    display(fig)
    save("./Output/dense_overflow_3d_james.pdf", fig, pt_per_unit=1)
    save("./Output/dense_overflow_3d_james.png", fig, px_per_unit=6)
end
#%%
