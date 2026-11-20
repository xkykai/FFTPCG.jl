using CairoMakie
using Makie

include("../benchmarks/rough_rayleigh_benard_setup.jl")

# Lengths are in units of the domain height.
triangle(η) = 8 - abs(mod(η, 16) - 8) # pyramid height in grid spacings, 16 spacings wide
pyramid(i, j) = min(triangle(i), triangle(j))
buoyancy(x, z, Lx) = -z + 1/2 + tanh((x - Lx / 2) / (Lx / 20)) / 4

colormap = :balance
levels = -0.75:0.125:0.75
azimuth = 1.45π
elevation = 0.12π
gray(name) = RGBf(Makie.to_color(name))

"Width and height of the box `Lx × Ly × Lz` on screen, seen from `azimuth` and `elevation` without perspective."
function projected_size(Lx, Ly, Lz; azimuth = azimuth, elevation = elevation)
    width = abs(sin(azimuth)) * Lx + abs(cos(azimuth)) * Ly
    height = cos(elevation) * Lz + abs(sin(elevation)) * (abs(cos(azimuth)) * Lx + abs(sin(azimuth)) * Ly)
    return width, height
end

"An `Axis3` showing `[0, Lx] × [0, Ly] × [0, Lz]` at `scale` points per unit length."
function scaled_axis3(position, title, Lx, Ly, Lz, scale; azimuth = azimuth, elevation = elevation)
    width, height = projected_size(Lx, Ly, Lz; azimuth, elevation) .* scale
    ax = Axis3(position; title, width, height, azimuth, elevation, perspectiveness = 0, aspect = :data,
               viewmode = :fitzoom, protrusions = (0, 0, 0, 12), titlesize = 9, titlealign = :left, titlegap = 2,
               halign = :left)
    hidedecorations!(ax)
    hidespines!(ax)
    limits!(ax, 0, Lx, 0, Ly, 0, Lz)
    return ax
end

"""
    block!(ax, x₀, x₁, Lx, Ly; shift)

One GPU's block `[x₀, x₁] × [0, Ly] × [0, 1]` of a domain of length `Lx`, drawn `shift` further along x, with filled
contours of the initial buoyancy on its front face and on its face at `x₀`.
"""
function block!(ax, x₀, x₁, Lx, Ly; shift)
    x = range(x₀, x₁, length = max(2, round(Int, 300 * (x₁ - x₀))))
    z = range(0, 1, length = 300)
    X = x .+ shift
    contourf!(ax, X, z, buoyancy.(x, z', Lx); colormap, levels, transformation = (:xz, 0))
    contourf!(ax, [0, Ly], z, repeat(buoyancy.(x₀, z', Lx), 2); colormap, levels, transformation = (:yz, x₀ + shift))
    X₀, X₁ = x₀ + shift, x₁ + shift
    mesh!(ax, Point3f[(X₀, 0, 1), (X₁, 0, 1), (X₁, Ly, 1), (X₀, Ly, 1)], [1 2 3; 1 3 4]; color = :gray85,
          shading = NoShading)
    for face in ([(X₀, 0, 0), (X₁, 0, 0), (X₁, 0, 1), (X₀, 0, 1)],
                 [(X₀, 0, 0), (X₀, Ly, 0), (X₀, Ly, 1), (X₀, 0, 1)],
                 [(X₀, 0, 1), (X₁, 0, 1), (X₁, Ly, 1), (X₀, Ly, 1)])
        lines!(ax, Point3f[face; face[1:1]]; color = :black, linewidth = 0.5)
    end
    return nothing
end

"A domain `[0, Lx] × [0, Ly] × [0, 1]` split into `ngpus` blocks along x, each pulled `gap` apart."
function domain!(ax, ngpus, Lx, Ly; gap)
    w = Lx / ngpus
    for r in ngpus:-1:1 # far to near for a camera on the -x, -y side
        block!(ax, (r - 1) * w, r * w, Lx, Ly; shift = (r - 1) * gap)
    end
    return nothing
end

"""
    walls!(ax; gap)

Solid cells of the bottom and top walls under 2 × 2 pyramids, `gap` grid spacings apart, in grid spacings.
"""
function walls!(ax; gap)
    n = 32
    H = [count(k -> k - 0.5 <= pyramid(i - 0.5, j - 0.5), 1:8) for i in 1:n, j in 1:n]
    height(i, j) = 1 <= i <= n && 1 <= j <= n ? H[i, j] : 0
    top = 16 + gap
    faces = [1 2 3; 1 3 4]
    quad!(points, shade) = mesh!(ax, Point3f.(points), faces; color = gray(shade), shading = NoShading)

    # Back to front, drawing only the faces in contact with the fluid
    a = ax.azimuth[]
    order = sort(vec(collect(Iterators.product(1:n, 1:n))), by = ((i, j),) -> cos(a) * i + sin(a) * j)
    for flip in (false, true), (i, j) in order
        x₀, x₁, y₀, y₁ = i - 1, i, j - 1, j
        z(h) = flip ? top - h : h
        h, hy, hx = H[i, j], height(i, j - 1), height(i - 1, j)
        flip || quad!([(x₀, y₀, h), (x₁, y₀, h), (x₁, y₁, h), (x₀, y₁, h)], :gray88)
        hy < h && quad!([(x₀, y₀, z(hy)), (x₁, y₀, z(hy)), (x₁, y₀, z(h)), (x₀, y₀, z(h))], :gray55)
        hx < h && quad!([(x₀, y₀, z(hx)), (x₀, y₁, z(hx)), (x₀, y₁, z(h)), (x₀, y₀, z(h))], :gray72)
    end
    quad!([(0, 0, top), (n, 0, top), (n, n, top), (0, n, top)], :gray88)
    return nothing
end

"Solid cells through a row of pyramid apexes next to the top and bottom walls, stacked at `position`."
function cells!(position, title)
    layout = GridLayout(position)
    i = 0:31
    k = 0:9
    top, bottom = map(1:2) do row
        ax = Axis(layout[row, 1]; aspect = DataAspect(), xticks = 0:8:32, xgridvisible = false, ygridvisible = false,
                  title = row == 1 ? title : "", titlesize = 9, titlealign = :left, titlegap = 2, width = 150)
        vlines!(ax, 0:32; color = :gray85, linewidth = 0.3)
        hlines!(ax, -10:10; color = :gray85, linewidth = 0.3)
        ax
    end
    solid = [k + 0.5 <= pyramid(i + 0.5, 7.5) ? 1.0 : NaN for i in i, k in k]
    heatmap!(top, i .+ 0.5, -k .- 0.5, solid; colormap = [:gray55, :gray55])
    heatmap!(bottom, i .+ 0.5, k .+ 0.5, solid; colormap = [:gray55, :gray55])
    lines!(top, 0:32, [-pyramid(i, 7.5) for i in 0:32]; color = :black, linewidth = 0.6)
    lines!(bottom, 0:32, [pyramid(i, 7.5) for i in 0:32]; color = :black, linewidth = 0.6)
    limits!(top, 0, 32, -10, 0)
    limits!(bottom, 0, 32, 0, 10)
    top.yticks = ([-8, 0], ["Nz − 8", "Nz"])
    bottom.yticks = [0, 8]
    top.ylabel = bottom.ylabel = "z / Δz"
    hidexdecorations!(top; ticks = false)
    bottom.xlabel = "x / Δx"
    rowgap!(layout, 4)
    return nothing
end

"The immersed walls in 3D and the solid cells through the pyramid apexes, with panels lettered from `letters`."
function wall_panels!(position, letters)
    layout = GridLayout(position)
    gap = 6
    ax = scaled_axis3(layout[1, 1], "($(letters[1])) Bottom and top walls, 2 × 2 pyramids", 32, 32, 16 + gap, 3.4;
                      azimuth = 1.32π, elevation = 0.1π)
    walls!(ax; gap)
    cells!(layout[1, 2], "($(letters[2])) Solid cells through the apexes")
    colgap!(layout, 20)
    return layout
end

buoyancy_colorbar(position; kw...) = Colorbar(position; colormap = cgrad(colormap, length(levels) - 1, categorical = true),
                                              limits = extrema(levels), label = "Initial buoyancy b", ticks = -0.5:0.5:0.5,
                                              kw...)

#%% Weak scaling on A100: one 448³ block per GPU stacked in x
fig = Figure(fontsize = 9, figure_padding = (4, 8, 4, 4))
scale = 48
gap = 0.15

small = GridLayout(fig[1, 1]; halign = :left)
for (n, letter, position) in ((1, 'a', small[1, 1]), (4, 'b', small[1, 2]), (8, 'c', fig[2, 1]))
    title = "($letter) $n GPU$(n > 1 ? "s" : "")"
    domain!(scaled_axis3(position, title, n + (n - 1) * gap, 1, 1, scale), n, n, 1; gap)
end
colgap!(small, 16)
buoyancy_colorbar(fig[1:2, 2]; width = 6, height = Relative(0.7))
wall_panels!(fig[3, 1:2], "de")
Label(fig[0, :], "Weak scaling: 448³ points per GPU, Lx = number of GPUs", font = :bold, fontsize = 9, halign = :left)
rowgap!(fig.layout, 8)
resize_to_layout!(fig)

save("./Output/weakscaling_setup.pdf", fig, pt_per_unit = 1)
save("./Output/weakscaling_setup.png", fig, px_per_unit = 6)

#%% Strong scaling on A100: the 448³ block split in x
fig = Figure(fontsize = 9, figure_padding = (4, 8, 4, 4))
scale = 105

for (col, (n, gap, letter)) in enumerate(zip((1, 4, 8), (0, 0.06, 0.04), "abc"))
    title = "($letter) $n GPU$(n > 1 ? "s" : "")" * (n > 1 ? ", $(448 ÷ n) × 448² each" : "")
    domain!(scaled_axis3(fig[1, col], title, 1 + (n - 1) * gap, 1, 1, scale), n, 1, 1; gap)
end
buoyancy_colorbar(fig[1, 4]; width = 6, height = Relative(0.8))
wall_panels!(fig[2, 1:4], "de")
Label(fig[0, :], "Strong scaling: 448³ points in total", font = :bold, fontsize = 9, halign = :left)
rowgap!(fig.layout, 8)
resize_to_layout!(fig)

save("./Output/strongscaling_setup.pdf", fig, pt_per_unit = 1)
save("./Output/strongscaling_setup.png", fig, px_per_unit = 6)

#%% One H100: squares Nx × Nx × 600 and strips Nx × 32 × 600, all with Δx = Δy = Δz = 1/600
fig = Figure(fontsize = 9, figure_padding = (4, 8, 4, 4))
scale = 100

squares = GridLayout(fig[1, 1])
for (col, (Nx, letter)) in enumerate(zip((128, 256, 512), "abc"))
    domain!(scaled_axis3(squares[1, col], "($letter) $(Nx)²", Nx / 600, Nx / 600, 1, scale), 1, Nx / 600, Nx / 600;
            gap = 0)
end
spacing = Axis(fig[1, 2]; title = "(d) Vertical grid spacing", titlesize = 9, titlealign = :left, titlegap = 2,
               xlabel = "Δz", ylabel = "z", xscale = log10, yticks = 0:0.5:1, xgridvisible = false, ygridvisible = false,
               xticks = ([2e-4, 5e-4, 1e-3, 2e-3], ["2×10⁻⁴", "5×10⁻⁴", "10⁻³", "2×10⁻³"]), height = 110)
for (grid, (Nz, color, side)) in zip(GRID_TYPES, ((600, 1, -1), (2400, 2, -1), (2400, 3, 1)))
    z = grid == "stretched" ? stretched_z_faces(Nz, 1) : range(0, 1, length = Nz + 1)
    Δz = diff(z)
    lines!(spacing, Δz, (z[1:end-1] .+ z[2:end]) ./ 2; color = Makie.wong_colors()[color])
    text!(spacing, maximum(Δz), 0.5; text = "$grid, Nz = $Nz", rotation = π / 2, fontsize = 7, offset = (2side, 0),
          align = (:center, side < 0 ? :bottom : :top), color = Makie.wong_colors()[color])
end
limits!(spacing, 1.5e-4, 2.5e-3, 0, 1)

strips = GridLayout(fig[2, 1:2])
for (col, (Nx, letter)) in enumerate(zip((512, 2048), "ef"))
    domain!(scaled_axis3(strips[1, col], "($letter) $Nx × 32", Nx / 600, 32 / 600, 1, scale), 1, Nx / 600, 32 / 600;
            gap = 0)
end
buoyancy_colorbar(fig[1:2, 3]; width = 6, height = Relative(0.6))
wall_panels!(fig[3, 1:3], "gh")
Label(fig[0, :], "One H100: squares Nx² × 600 and strips Nx × 32 × 600 at Δx = Δy = 1/600", font = :bold,
      fontsize = 9, halign = :left)
rowgap!(fig.layout, 8)
resize_to_layout!(fig)

save("./Output/single_H100_setup.pdf", fig, pt_per_unit = 1)
save("./Output/single_H100_setup.png", fig, px_per_unit = 6)
#%%
