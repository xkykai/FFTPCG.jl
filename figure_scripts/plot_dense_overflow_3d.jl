using Oceananigans
using Oceananigans.Units
using Oceananigans.ImmersedBoundaries: mask_immersed_field!
using GLMakie

#%%
filename = "dense_overflow_Nx_1536_Ny_640_Nz_128_CG_nu4h_5.0e6_nu4z_80.0_14days"
snapshot_time = 13days
plume_threshold = 0.05
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

# `φ(i, j, k)` where the tracer falls below `threshold`, NaN where it never exceeds it
function at_tracer_surface(φ, threshold)
    values = fill(NaN, Nx, Ny)

    for j in 1:Ny, i in 1:Nx
        k = findlast(>(threshold), view(c, i, j, :))
        isnothing(k) && continue
        above = k < Nz ? (c[i, j, k] - threshold) / (c[i, j, k] - c[i, j, k+1]) : 0
        values[i, j] = φ(i, j, k) + above * (φ(i, j, min(k + 1, Nz)) - φ(i, j, k))
    end

    return values
end

tracer_surface(threshold) = max.(at_tracer_surface((i, j, k) -> z[k], threshold), bottom_height)
plume_top = tracer_surface(plume_threshold)

# Contours of `buoyancy` as lines lying on the surface `height`
function buoyancy_contours(height, buoyancy=bottom(b))
    points = Point3f[]
    contours = Makie.Contours.contours(x, y, buoyancy, -0.018:0.0015:0)

    for level in Makie.Contours.levels(contours), line in Makie.Contours.lines(level)
        for (x_point, y_point) in zip(Makie.Contours.coordinates(line)...)
            i, j = clamp(searchsortedlast(x, x_point), 1, Nx), clamp(searchsortedlast(y, y_point), 1, Ny)
            push!(points, Point3f(x_point, y_point, height[i, j] + 0.02))
        end

        push!(points, Point3f(NaN))
    end

    return points
end

# Tracer colors with opacity equal to `tracer`, dimmed by the tracer lying between each voxel and
# the light and by the angle between the tracer gradient and the light
function lit_voxels(tracer; light_step=(1, -1), extinction=0.12, ambient=0.25)
    nx, ny, nz = size(tracer)
    toward_light = (light_step..., 1) ./ hypot(light_step..., 1)
    shadow = zeros(nx, ny, nz)
    voxels = Array{RGBAf}(undef, nx, ny, nz)

    for k in nz-1:-1:1, j in 1:ny, i in 1:nx
        i_light, j_light = clamp(i + light_step[1], 1, nx), clamp(j + light_step[2], 1, ny)
        shadow[i, j, k] = shadow[i_light, j_light, k+1] + tracer[i_light, j_light, k+1]
    end

    for k in 1:nz, j in 1:ny, i in 1:nx
        gradient = (tracer[min(i+1, nx), j, k] - tracer[max(i-1, 1), j, k],
                    tracer[i, min(j+1, ny), k] - tracer[i, max(j-1, 1), k],
                    tracer[i, j, min(k+1, nz)] - tracer[i, j, max(k-1, 1)])

        steepness = hypot(gradient...)
        facing = steepness > 0 ? max(-sum(gradient .* toward_light) / steepness, 0) : 1
        weight = min(steepness / 0.2, 1)
        brightness = (1 - weight + weight * (0.45 + 0.55 * facing)) * (ambient + (1 - ambient) * exp(-extinction * shadow[i, j, k]))
        color = tracer_color(tracer[i, j, k])
        voxels[i, j, k] = RGBAf(color.r * brightness, color.g * brightness, color.b * brightness, tracer[i, j, k])
    end

    return voxels
end

#%%
tracer_kw = (; colormap=:matter, colorrange=(0, 1))
tracer_colors = Makie.to_colormap(:matter)
tracer_color(tracer) = tracer_colors[round(Int, 1 + 255 * clamp(tracer, 0, 1))]
with_alpha(color, alpha) = RGBAf(color.r, color.g, color.b, alpha)

function overflow_axis()
    fig = Figure(size=(1500, 760), fontsize=20)
    ax = Axis3(fig[1, 1];
               xlabel = "x (km)",
               ylabel = "y (km)",
               zlabel = "z (km)",
               aspect = (1, grid.Ly / grid.Lx, 0.18),
               limits = (extrema(x), extrema(y), (-3.3, 0)),
               azimuth = -0.6π,
               elevation = 0.24π,
               perspectiveness = 0.3,
               protrusions = 70,
               xlabeloffset = 60,
               ylabeloffset = 70,
               zlabeloffset = 70,
               xgridvisible = false,
               ygridvisible = false,
               zgridvisible = false)

    surface!(ax, x, y, bottom_height; color=zeros(Nx, Ny), colormap=[:gray78, :gray78])
    return fig, ax
end

tracer_colorbar!(fig) = Colorbar(fig[1, 2]; tracer_kw..., label="Tracer concentration", height=Relative(0.5))

#%% Plume surface as shaded relief, colored and contoured by the buoyancy on it
let
    fig, ax = overflow_axis()
    plume_buoyancy = at_tracer_surface((i, j, k) -> b[i, j, k], plume_threshold)
    slope_buoyancy = ifelse.(isnan.(plume_top), bottom(b), NaN)

    plume = surface!(ax, x, y, plume_top; color=plume_buoyancy .* 1e3, colormap=:haline, colorrange=(-14, 0))
    lines!(ax, buoyancy_contours(plume_top, plume_buoyancy); color=:black, linewidth=1.5)
    lines!(ax, buoyancy_contours(bottom_height, slope_buoyancy); color=:black, linewidth=1.5)
    Colorbar(fig[1, 2], plume; height=Relative(0.5),
             label=rich("Buoyancy (10", superscript("−3"), " m s", superscript("−2"), ")"))

    save("./Output/dense_overflow_3d_relief.png", fig, px_per_unit=2)
end

#%% Translucent plume surface above the bottom tracer and buoyancy contours
let
    fig, ax = overflow_axis()
    tracer_colorbar!(fig)
    sea_floor = ifelse.(wet, bottom_height .+ 0.004, NaN)

    surface!(ax, x, y, sea_floor; color=bottom(c), tracer_kw..., shading=NoShading)
    lines!(ax, buoyancy_contours(bottom_height); color=:black, linewidth=1.5)
    surface!(ax, x, y, plume_top; color=bottom(c), colormap=with_alpha.(tracer_colors, 0.45), colorrange=(0, 1), transparency=true)

    save("./Output/dense_overflow_3d_glass.png", fig, px_per_unit=2)
end

#%% Nested surfaces of constant tracer, each in the color of its tracer value
let
    fig, ax = overflow_axis()
    tracer_colorbar!(fig)
    lines!(ax, buoyancy_contours(bottom_height); color=:gray25, linewidth=1.5)

    for (tracer, alpha) in ((0.7, 1), (0.4, 0.35), (0.1, 0.15))
        color = with_alpha(tracer_color(tracer), alpha)
        surface!(ax, x, y, tracer_surface(tracer); color=zeros(Nx, Ny), colormap=[color, color], transparency=alpha < 1)
    end

    save("./Output/dense_overflow_3d_isosurfaces.png", fig, px_per_unit=2)
end

#%% Volume rendering of the tracer, with opacity proportional to concentration
let
    fig, ax = overflow_axis()
    tracer_colorbar!(fig)
    voxel(tracer) = with_alpha(tracer_color(tracer), clamp(tracer, 0, 1))
    voxels = [isnan(tracer) ? RGBAf(0, 0, 0, 0) : voxel(tracer) for tracer in view(c, 1:3:Nx, 1:3:Ny, :)]

    lines!(ax, buoyancy_contours(bottom_height); color=:gray25, linewidth=1.5)
    volume!(ax, x[1] .. x[end], y[1] .. y[end], z[1] .. z[end], voxels; algorithm=:absorptionrgba, absorption=70)

    save("./Output/dense_overflow_3d_volume.png", fig, px_per_unit=2)
end

#%% Volume rendering lit from the east, with self-shadowing
let
    fig, ax = overflow_axis()
    tracer_colorbar!(fig)
    tracer = clamp.(replace(c[1:3:Nx, 1:3:Ny, :], NaN => 0), 0, 1)

    lines!(ax, buoyancy_contours(bottom_height); color=:gray25, linewidth=1.5)
    volume!(ax, x[1] .. x[end], y[1] .. y[end], z[1] .. z[end], lit_voxels(tracer); algorithm=:absorptionrgba, absorption=70)

    save("./Output/dense_overflow_3d_volume_lit.png", fig, px_per_unit=2)
end
#%%
