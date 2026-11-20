using CairoMakie
using Makie

include("../benchmarks/rough_rayleigh_benard_setup.jl")

# Lengths are in units of the domain height; Δ is the grid spacing, 1/448 on A100 and 1/600 on H100.
triangle(η, Δ) = 8Δ - abs(mod(η, 16Δ) - 8Δ)
pyramid(x, y, Δ) = min(triangle(x, Δ), triangle(y, Δ))
buoyancy(x, z, Lx) = -z + 1/2 + tanh((x - Lx / 2) / (Lx / 20)) / 4

# x-z slices pass through a row of pyramid apexes
solid(x, z, Δ) = (z <= pyramid(x, 7.5Δ, Δ)) | (z >= 1 - pyramid(x, 7.5Δ, Δ))

colormap = :balance
levels = -0.75:0.125:0.75
azimuth = 1.43π
elevation = 0.25π
front_offset = 0.9 # distance the front faces are pulled forward, in units of the domain height drawn

"Square pyramids on `[0, Lx] × [0, Ly]` at `o`, their faces shaded by the way they face."
function pyramids!(ax, o, Lx, Ly, Δ)
    h = 8Δ
    points = Point3f[]
    colors = RGBf[]
    shades = RGBf.(Makie.to_color.([:gray70, :gray50, :gray40, :gray88])) # -y, +x, +y and -x faces
    for a in 0:2h:Lx-h, c in 0:2h:Ly-h
        apex = o + Point3f(a + h, c + h, h)
        corners = [o + Point3f(p) for p in ((a, c, 0), (a + 2h, c, 0), (a + 2h, c + 2h, 0), (a, c + 2h, 0))]
        for k in 1:4
            append!(points, (corners[k], corners[mod1(k + 1, 4)], apex))
            append!(colors, fill(shades[k], 3))
        end
    end
    mesh!(ax, points, reshape(1:length(points), 3, :)'; color = colors, shading = NoShading, rasterize = 5)
    return nothing
end

"Edges of the box `[0, Lx] × [0, Ly] × [0, Lz]` at `o`."
function edges!(ax, o, Lx, Ly, Lz)
    corners = [(0, 0), (Lx, 0), (Lx, Ly), (0, Ly)]
    for (x, y) in corners
        lines!(ax, [o + Point3f(x, y, 0), o + Point3f(x, y, Lz)]; color = :black, linewidth = 0.5)
    end
    for z in (0, Lz)
        lines!(ax, [o + Point3f(x, y, z) for (x, y) in [corners; corners[1:1]]]; color = :black, linewidth = 0.5)
    end
    return nothing
end

"""
    front!(ax, o, x₀, x₁, Lx, Lz, Δ; y)

Filled contours of the initial buoyancy on an x-z slice through `[x₀, x₁] × [0, Lz]` of a domain of length `Lx`,
drawn at `o` on the plane `y`, with the immersed walls in grey.
"""
function front!(ax, o, x₀, x₁, Lx, Lz, Δ; y)
    x = range(x₀ + Δ / 2, x₁ - Δ / 2, step = Δ)
    z = range(Δ / 2, Lz - Δ / 2, step = Δ)
    b = [solid(x, z, Δ) ? NaN : buoyancy(x, z, Lx) for x in x, z in z]
    walls = [solid(x, z, Δ) ? 1.0 : NaN for x in x, z in z]
    X = x .- x₀ .+ o[1]
    Z = z .+ o[3]
    transformation = (:xz, o[2] + y)
    contourf!(ax, X, Z, b; colormap, levels, transformation, rasterize = 5)
    heatmap!(ax, X, Z, walls; colormap = [:gray40, :gray40], transformation, rasterize = 5)
    edges!(ax, Point3f(o[1], o[2] + y, o[3]), x₁ - x₀, 0, Lz)
    return nothing
end

"""
    domain!(ax, o, ngpus, Lx, Ly, Δ; gap, Lz = 1)

A domain `[0, Lx] × [0, Ly] × [0, Lz]` at `o` split into `ngpus` blocks along x, each pulled `gap` apart, with the
front face of each block pulled forward to show the buoyancy front. Returns the length drawn along x.
"""
function domain!(ax, o, ngpus, Lx, Ly, Δ; gap, Lz = 1)
    w = Lx / ngpus
    corner(r) = o + Point3f((r - 1) * (w + gap), 0, 0)

    # Far to near for a camera on the -x, -y side
    for r in ngpus:-1:1
        pyramids!(ax, corner(r), w, Ly, Δ)
        edges!(ax, corner(r), w, Ly, Lz)
    end
    for r in ngpus:-1:1
        front!(ax, corner(r), (r - 1) * w, r * w, Lx, Lz, Δ; y = -front_offset * Lz)
    end
    return Lx + (ngpus - 1) * gap
end

function axis3(position; kw...)
    ax = Axis3(position; azimuth, elevation, perspectiveness = 0.2, aspect = :data, viewmode = :fitzoom,
               protrusions = (0, 0, 0, 12), titlesize = 9, titlegap = 0, kw...)
    hidedecorations!(ax)
    hidespines!(ax)
    return ax
end

label!(ax, p, text; align = (:left, :top)) = text!(ax, p; text, align, fontsize = 8, clip_planes = Plane3f[])

"The immersed bottom wall under 2 × 2 pyramids as columns of solid cells, in units of grid spacings."
function stepped_pyramids!(ax)
    n = 32
    H = [count(k -> k - 0.5 <= pyramid(i - 0.5, j - 0.5, 1), 1:8) for i in 1:n, j in 1:n]
    shades = RGBf.(Makie.to_color.([:gray88, :gray70, :gray80])) # top, -y and -x faces
    faces = [1 2 3; 1 3 4]

    # Back to front for a camera on the -x, -y side
    for (i, j) in sort(vec(collect(Iterators.product(1:n, 1:n))), by = ((i, j),) -> cos(azimuth) * i + sin(azimuth) * j)
        z = H[i, j]
        x₀, x₁, y₀, y₁ = i - 1, i, j - 1, j
        top = Point3f.([(x₀, y₀, z), (x₁, y₀, z), (x₁, y₁, z), (x₀, y₁, z)])
        front = Point3f.([(x₀, y₀, 0), (x₁, y₀, 0), (x₁, y₀, z), (x₀, y₀, z)])
        side = Point3f.([(x₀, y₀, 0), (x₀, y₁, 0), (x₀, y₁, z), (x₀, y₀, z)])
        for (quad, shade) in zip((top, front, side), shades)
            mesh!(ax, quad, faces; color = shade, shading = NoShading)
        end
    end
    limits!(ax, 0, n, 0, n, 0, 8)
    return nothing
end

"Grid cells on the bottom wall at the slice: each pyramid is 16 cells wide and 8 tall."
function cells!(ax)
    i = 0:31
    k = 0:9
    cells = [k + 0.5 <= pyramid(i + 0.5, 7.5, 1) ? 1.0 : NaN for i in i, k in k]
    heatmap!(ax, i .+ 0.5, k .+ 0.5, cells; colormap = [:gray55, :gray55])
    vlines!(ax, 0:32; color = :gray80, linewidth = 0.3)
    hlines!(ax, 0:10; color = :gray80, linewidth = 0.3)
    lines!(ax, 0:32, [pyramid(i, 7.5, 1) for i in 0:32]; color = :black, linewidth = 0.6)
    limits!(ax, 0, 32, 0, 10)
    return nothing
end

function pyramid_panels!(position, letters)
    layout = GridLayout(position)
    stepped_pyramids!(axis3(layout[1, 1]; title = "($(letters[1])) Immersed bottom wall", elevation = 0.15π))
    ax = Axis(layout[1, 2]; aspect = DataAspect(), xlabel = "x / Δx", ylabel = "z / Δz", xticks = 0:8:32, yticks = 0:4:8,
              title = "($(letters[2])) Solid cells through the apexes", titlesize = 9, titlegap = 2)
    cells!(ax)
    colsize!(layout, 1, Auto(1.4))
    return nothing
end

buoyancy_colorbar(position; kw...) = Colorbar(position; colormap = cgrad(colormap, length(levels) - 1, categorical = true),
                                              limits = extrema(levels), label = "Initial buoyancy b", ticks = -0.5:0.5:0.5,
                                              width = 6, kw...)

#%% Weak scaling on A100: one 448³ block per GPU stacked in x
fig = Figure(size = (510, 420), fontsize = 9, figure_padding = (2, 8, 2, 4))

ax = axis3(fig[1, 1]; title = "(a) Weak scaling, 448³ points per GPU")
for (row, n) in enumerate((1, 4, 8))
    o = Point3f(0, 0, -3.2 * (row - 1))
    domain!(ax, o, n, n, 1, 1 / 448; gap = 0.3)
    label!(ax, o + Point3f(-0.3, -front_offset, 0.5), n == 1 ? "1 GPU" : "$n GPUs"; align = (:right, :center))
end
limits!(ax, -1.5, 8 + 7 * 0.3, -front_offset, 1, -6.4, 1)
buoyancy_colorbar(fig[2, 1]; vertical = false, flipaxis = false, width = Relative(0.4), height = 6)
pyramid_panels!(fig[3, 1], "bc")
rowsize!(fig.layout, 3, Relative(0.25))

save("./Output/weakscaling_setup.pdf", fig, pt_per_unit = 1)
save("./Output/weakscaling_setup.png", fig, px_per_unit = 6)

#%% Strong scaling on A100: the 448³ block split in x
fig = Figure(size = (510, 320), fontsize = 9, figure_padding = (2, 8, 2, 4))

ax = axis3(fig[1, 1]; title = "(a) Strong scaling, 448³ points in total")
x = 0.0
for n in (1, 4, 8)
    o = Point3f(x, 0, 0)
    global x += domain!(ax, o, n, 1, 1, 1 / 448; gap = 0.08) + 0.6
    label!(ax, o + Point3f(0, -front_offset, -0.05), n == 1 ? "1 GPU" : "$n GPUs, $(448 ÷ n) × 448² each")
end
limits!(ax, 0, x - 0.6, -front_offset, 1, -0.6, 1)
buoyancy_colorbar(fig[2, 1]; vertical = false, flipaxis = false, width = Relative(0.4), height = 6)
pyramid_panels!(fig[3, 1], "bc")
rowsize!(fig.layout, 3, Relative(0.35))

save("./Output/strongscaling_setup.pdf", fig, pt_per_unit = 1)
save("./Output/strongscaling_setup.png", fig, px_per_unit = 6)

#%% One H100: squares Nx × Nx × 600 and strips Nx × 32 × 600 at the same grid spacing, bottom 48 points in z
fig = Figure(size = (510, 260), fontsize = 9, figure_padding = (2, 8, 2, 4))

Δ = 1 / 600
Lz = 48Δ
ax = axis3(fig[1, 1]; title = "(a) Bottom 48 of 600 points in z, Δx = Δy = 1/600")
x = 0.0
for Nx in (32, 64, 128)
    o = Point3f(x, 0, 0.3)
    global x += domain!(ax, o, 1, Nx * Δ, Nx * Δ, Δ; gap = 0, Lz) + 40Δ
    label!(ax, o + Point3f(0, Nx * Δ, Lz), "$Nx × $Nx"; align = (:left, :bottom))
end
x = 0.0
for Nx in (128, 512)
    o = Point3f(x, 0, 0)
    global x += domain!(ax, o, 1, Nx * Δ, 32Δ, Δ; gap = 0, Lz) + 40Δ
    label!(ax, o + Point3f(0, 32Δ, Lz), "$Nx × 32"; align = (:left, :bottom))
end
limits!(ax, 0, x - 40Δ, -front_offset * Lz, 128Δ, 0, 0.3 + Lz)

spacing = Axis(fig[1, 2]; title = "(b) Vertical grid spacing", titlesize = 9, titlegap = 2, xlabel = "Δz", ylabel = "z",
               xscale = log10, yticks = 0:0.5:1, xgridvisible = false, ygridvisible = false,
               xticks = ([2e-4, 5e-4, 1e-3, 2e-3], ["2×10⁻⁴", "5×10⁻⁴", "10⁻³", "2×10⁻³"]))
for (grid, (Nz, color)) in zip(GRID_TYPES, ((600, 1), (2400, 2), (2400, 3)))
    z = grid == "stretched" ? stretched_z_faces(Nz, 1) : range(0, 1, length = Nz + 1)
    Δz = diff(z)
    lines!(spacing, Δz, (z[1:end-1] .+ z[2:end]) ./ 2; color = Makie.wong_colors()[color])
    text!(spacing, maximum(Δz), 0.5; text = grid, rotation = π / 2, align = (:center, :bottom), fontsize = 7,
          offset = (-2, 0), color = Makie.wong_colors()[color])
end
limits!(spacing, 1.5e-4, 2.5e-3, 0, 1)
buoyancy_colorbar(fig[2, 1]; vertical = false, flipaxis = false, width = Relative(0.5), height = 6)
colsize!(fig.layout, 2, Relative(0.3))

save("./Output/single_H100_setup.pdf", fig, pt_per_unit = 1)
save("./Output/single_H100_setup.png", fig, px_per_unit = 6)
#%%
