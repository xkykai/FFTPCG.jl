using Oceananigans
using Oceananigans.Units
using Oceananigans.ImmersedBoundaries: mask_immersed_field!
using GLMakie

#%%
filename = "dense_overflow_Nx_1536_Ny_640_Nz_128_CG_nu4h_5.0e6_nu4z_80.0_14days"
snapshot_time = 13days
hₑ = 600    # embayment depth (m)
tanα = 0.01 # bottom slope

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
x, y, z = nodes(grid, Center(), Center(), Center()) ./ kilometers
Δz = grid.Lz / Nz / kilometers

bottom_k = [something(findfirst(!isnan, view(c, i, j, :)), 0) for i in 1:Nx, j in 1:Ny]
wet = bottom_k .> 0
bottom(φ) = [wet[i, j] ? φ[i, j, bottom_k[i, j]] : NaN32 for i in 1:Nx, j in 1:Ny]
bottom_height = [wet[i, j] ? -hₑ / kilometers + tanα * min(y[j], 0) : 0.0 for i in 1:Nx, j in 1:Ny]

# Height (km) at which the tracer falls below `threshold`, NaN outside the plume
function plume_top(threshold)
    top = fill(NaN, Nx, Ny)

    for j in 1:Ny, i in 1:Nx
        k = findlast(>(threshold), view(c, i, j, :))
        isnothing(k) && continue
        above = k < Nz ? (c[i, j, k] - threshold) / (c[i, j, k] - c[i, j, k+1]) : 0
        top[i, j] = max(z[k] + above * Δz, bottom_height[i, j])
    end

    return top
end

# Contours of bottom buoyancy as lines lying on the surface `height`
function buoyancy_contours(height)
    points = Point3f[]
    contours = Makie.Contours.contours(x, y, replace(bottom(b), NaN => 1), -0.018:0.0015:-0.003)

    for level in Makie.Contours.levels(contours), line in Makie.Contours.lines(level)
        for (x_point, y_point) in zip(Makie.Contours.coordinates(line)...)
            i, j = clamp(searchsortedlast(x, x_point), 1, Nx), clamp(searchsortedlast(y, y_point), 1, Ny)
            push!(points, Point3f(x_point, y_point, height[i, j] + 0.02))
        end

        push!(points, Point3f(NaN))
    end

    return points
end

#%%
tracer_kw = (; colormap=:matter, colorrange=(0, 1))
tracer_color(tracer, alpha) = (color = cgrad(:matter)[tracer]; RGBAf(color.r, color.g, color.b, alpha))
plume_colormap = [tracer_color(tracer, tracer > 0.03) for tracer in range(0, 1, length=256)]

function overflow_axis(; zlabel, zlimits, zaspect, azimuth=-0.6π, elevation)
    fig = Figure(size=(1500, 760), fontsize=20)
    ax = Axis3(fig[1, 1]; zlabel, azimuth, elevation,
               xlabel = "x (km)",
               ylabel = "y (km)",
               aspect = (1, grid.Ly / grid.Lx, zaspect),
               limits = (extrema(x), extrema(y), zlimits),
               perspectiveness = 0.3,
               protrusions = 70,
               xlabeloffset = 60,
               ylabeloffset = 70,
               zlabeloffset = 70,
               xgridvisible = false,
               ygridvisible = false,
               zgridvisible = false)

    Colorbar(fig[1, 2]; tracer_kw..., label="Tracer concentration", height=Relative(0.5))
    return fig, ax
end

gray_surface!(ax, height) = surface!(ax, x, y, height; color=zeros(Nx, Ny), colormap=[:gray78, :gray78])

true_depth = (; zlabel="z (km)", zlimits=(-3.3, 0), zaspect=0.18, elevation=0.24π)
above_bottom = (; zlabel="Height above bottom (km)", zlimits=(0, 1.2), zaspect=0.16)
flat_bottom = ifelse.(wet, 0.0, NaN)
tops = Dict(threshold => plume_top(threshold) for threshold in (0.05, 0.5, 0.8))

#%% Plume surface as shaded relief with bottom buoyancy contours
let
    fig, ax = overflow_axis(; true_depth...)
    relief = ifelse.(isnan.(tops[0.05]), bottom_height, tops[0.05])

    gray_surface!(ax, bottom_height)
    surface!(ax, x, y, tops[0.05]; color=bottom(c), tracer_kw...)
    lines!(ax, buoyancy_contours(relief); color=:black, linewidth=1.5)

    save("./Output/dense_overflow_3d_relief.png", fig, px_per_unit=2)
end

#%% Nested tracer shells above a flattened bottom
let
    fig, ax = overflow_axis(; above_bottom..., elevation=0.16π)

    gray_surface!(ax, zeros(Nx, Ny))
    surface!(ax, x, y, flat_bottom .+ 0.002; color=bottom(c), tracer_kw..., shading=NoShading)
    lines!(ax, buoyancy_contours(flat_bottom); color=:black, linewidth=1.5)

    for (threshold, alpha) in ((0.8, 1), (0.5, 0.55), (0.05, 0.3))
        color = tracer_color(threshold, alpha)
        surface!(ax, x, y, tops[threshold] .- bottom_height;
                 color=zeros(Nx, Ny), colormap=[color, color], transparency=alpha < 1)
    end

    save("./Output/dense_overflow_3d_shells.png", fig, px_per_unit=2)
end

#%% Tracer sections every 50 km along the slope, above a flattened bottom
let
    fig, ax = overflow_axis(; above_bottom..., azimuth=-0.78π, elevation=0.14π)
    cells = 1:round(Int, 1.2 / Δz)
    height = (cells .- 0.5) .* Δz

    gray_surface!(ax, zeros(Nx, Ny))
    surface!(ax, x, y, flat_bottom .+ 0.002; color=bottom(c), tracer_kw..., shading=NoShading)
    lines!(ax, buoyancy_contours(flat_bottom); color=:black, linewidth=1.5)

    for x_section in -600:50:-50
        i = searchsortedlast(x, x_section)
        section = [get(view(c, i, j, :), bottom_k[i, j] + m - 1, NaN32) for j in 1:Ny, m in cells]
        surface!(ax, fill(x[i], Ny, length(cells)), repeat(y, 1, length(cells)), repeat(height', Ny, 1);
                 color=section, colormap=plume_colormap, colorrange=(0, 1), shading=NoShading, transparency=true)
    end

    save("./Output/dense_overflow_3d_sections.png", fig, px_per_unit=2)
end

#%% Maximum tracer along each line of sight
let
    fig, ax = overflow_axis(; true_depth...)

    gray_surface!(ax, bottom_height)
    lines!(ax, buoyancy_contours(bottom_height); color=:gray30, linewidth=1.5)
    volume!(ax, x[1] .. x[end], y[1] .. y[end], z[1] .. z[end], replace(c[1:2:end, 1:2:end, :], NaN => 0);
            algorithm=:mip, colormap=plume_colormap, colorrange=(0, 1))

    save("./Output/dense_overflow_3d_projection.png", fig, px_per_unit=2)
end
#%%
