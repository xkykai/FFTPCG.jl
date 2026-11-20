using CairoMakie
using Makie
using LinearAlgebra: ⋅

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
               halign = :left, valign = :bottom)
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

Solid cells of the bottom and top walls under 2 × 2 pyramids, `gap` grid spacings apart, in grid spacings. The top of
the top wall is see-through.
"""
function walls!(ax; gap)
    n = 32
    H = [count(k -> k - 0.5 <= pyramid(i - 0.5, j - 0.5), 1:8) for i in 1:n, j in 1:n]
    height(i, j) = 1 <= i <= n && 1 <= j <= n ? H[i, j] : 0
    top = 16 + gap

    # Faces in contact with the fluid
    quads = Tuple{Vector{Point3f}, Symbol}[]
    for flip in (false, true), i in 1:n, j in 1:n
        x₀, x₁, y₀, y₁ = i - 1, i, j - 1, j
        z(h) = flip ? top - h : h
        h = H[i, j]
        push!(quads, (Point3f[(x₀, y₀, z(h)), (x₁, y₀, z(h)), (x₁, y₁, z(h)), (x₀, y₁, z(h))], flip ? :gray80 : :gray88))
        for (hn, side, shade) in ((height(i, j - 1), [(x₀, y₀), (x₁, y₀)], :gray55), (height(i - 1, j), [(x₀, y₀), (x₀, y₁)], :gray72),
                                  (height(i, j + 1), [(x₀, y₁), (x₁, y₁)], :gray62), (height(i + 1, j), [(x₁, y₀), (x₁, y₁)], :gray66))
            hn < h || continue
            (a, b), (c, d) = side
            push!(quads, (Point3f[(a, b, z(hn)), (c, d, z(hn)), (c, d, z(h)), (a, b, z(h))], shade))
        end
    end

    # Far to near along the line of sight
    e, a = ax.elevation[], ax.azimuth[]
    view = Point3f(cos(e) * cos(a), cos(e) * sin(a), sin(e))
    faces = [1 2 3; 1 3 4]
    for (points, shade) in sort(quads, by = ((points, _),) -> sum(points) ⋅ view)
        mesh!(ax, points, faces; color = gray(shade), shading = NoShading)
    end
    lid = Point3f[(0, 0, top), (n, 0, top), (n, n, top), (0, n, top)]
    mesh!(ax, lid, faces; color = (:gray70, 0.25), shading = NoShading)
    lines!(ax, [lid; lid[1:1]]; color = :gray40, linewidth = 0.5)
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

"The immersed walls under 2 × 2 pyramids in 3D and in an x-z cross-section, with panels lettered from `letters`."
function wall_panels!(position, letters)
    layout = GridLayout(position)
    gap = 6
    Label(layout[0, 1:2], "Immersed bottom and top walls, 2 × 2 pyramids"; font = :bold, halign = :left)
    ax = scaled_axis3(layout[1, 1], "($(letters[1])) 3D view, top wall see-through", 32, 32, 16 + gap, 3.4;
                      azimuth = 1.32π, elevation = 0.12π)
    walls!(ax; gap)
    cells!(layout[1, 2], "($(letters[2])) x–z cross-section through the apexes")
    colgap!(layout, 20)
    rowgap!(layout, 2)
    return layout
end

buoyancy_colorbar(position; kw...) = Colorbar(position; colormap = cgrad(colormap, length(levels) - 1, categorical = true),
                                              limits = extrema(levels), label = "Initial buoyancy b", ticks = -0.5:0.5:0.5,
                                              kw...)

#%% Weak scaling on A100: one 448³ block per GPU stacked in x
fig = Figure(fontsize = 9, figure_padding = (4, 8, 4, 4))
scale = 48
gap = 0.15

small = GridLayout(fig[1, 1]; halign = :center)
for (n, letter, position) in ((1, 'a', small[1, 1]), (4, 'b', small[1, 2]), (8, 'c', fig[2, 1]))
    title = "($letter) $n GPU$(n > 1 ? "s" : ""), " * (n == 1 ? "448³" : "$(448n) × 448²")
    ax = scaled_axis3(position, title, n + (n - 1) * gap, 1, 1, scale)
    ax.halign = :center
    domain!(ax, n, n, 1; gap)
end
colgap!(small, 30)
buoyancy_colorbar(fig[1:2, 2]; width = 6, height = Relative(0.7))
wall_panels!(fig[3, 1:2], "de")
rowgap!(fig.layout, 8)
resize_to_layout!(fig)

save("./Output/weakscaling_setup.pdf", fig, pt_per_unit = 1)
save("./Output/weakscaling_setup.png", fig, px_per_unit = 6)

#%% Strong scaling on A100: the 448³ block split in x
fig = Figure(fontsize = 9, figure_padding = (4, 8, 4, 4))
scale = 105

for (col, (n, gap, letter)) in enumerate(zip((1, 4, 8), (0, 0.06, 0.04), "abc"))
    title = "($letter) $n GPU$(n > 1 ? "s" : ""), " * (n == 1 ? "448³" : "$(448 ÷ n) × 448² each")
    domain!(scaled_axis3(fig[1, col], title, 1 + (n - 1) * gap, 1, 1, scale), n, 1, 1; gap)
end
buoyancy_colorbar(fig[1, 4]; width = 6, height = Relative(0.8))
wall_panels!(fig[2, 1:4], "de")
rowgap!(fig.layout, 8)
resize_to_layout!(fig)

save("./Output/strongscaling_setup.pdf", fig, pt_per_unit = 1)
save("./Output/strongscaling_setup.png", fig, px_per_unit = 6)

#%% One H100: squares and strips with Δx = Δy = 1/600 and 600 points in z
fig = Figure(fontsize = 9, figure_padding = (4, 8, 4, 4))
scale = 100

squares = GridLayout(fig[1, 1]; halign = :left, valign = :bottom)
Label(squares[0, 1:3], "Squares, (Nx, Ny, Nz) = (N, N, 600)"; font = :bold, halign = :left)
for (col, (N, letter)) in enumerate(zip((128, 256, 512), "abc"))
    ax = scaled_axis3(squares[1, col], "($letter) N = $N", N / 600, N / 600, 1, scale; azimuth = 1.3π, elevation = 0.2π)
    domain!(ax, 1, N / 600, N / 600; gap = 0)
end
colgap!(squares, 24)
rowgap!(squares, 2)
spacing = Axis(fig[1, 2]; title = "(d) Vertical grid spacing", titlesize = 9, titlealign = :left, titlegap = 2,
               xlabel = "Δz", ylabel = "z", xscale = log10, yticks = 0:0.5:1, xgridvisible = false, ygridvisible = false,
               xticks = ([2e-4, 5e-4, 1e-3, 2e-3], ["2×10⁻⁴", "5×10⁻⁴", "10⁻³", "2×10⁻³"]), height = 130)
for (grid, (Nz, color, side)) in zip(GRID_TYPES, ((600, 1, -1), (2400, 2, -1), (2400, 3, 1)))
    z = grid == "stretched" ? stretched_z_faces(Nz, 1) : range(0, 1, length = Nz + 1)
    Δz = diff(z)
    lines!(spacing, Δz, (z[1:end-1] .+ z[2:end]) ./ 2; color = Makie.wong_colors()[color])
    text!(spacing, maximum(Δz), 0.5; text = "$grid, Nz = $Nz", rotation = π / 2, fontsize = 9, offset = (2side, 0),
          align = (:center, side < 0 ? :bottom : :top), color = Makie.wong_colors()[color])
end
limits!(spacing, 1.5e-4, 2.5e-3, 0, 1)

strips = GridLayout(fig[2, 1:2]; halign = :left)
Label(strips[0, 1:2], "Strips, (Nx, Ny, Nz) = (Nx, 32, 600)"; font = :bold, halign = :left)
for (col, (Nx, letter)) in enumerate(zip((512, 2048), "ef"))
    ax = scaled_axis3(strips[1, col], "($letter) (Nx, Ny) = ($Nx, 32)", Nx / 600, 32 / 600, 1, scale)
    domain!(ax, 1, Nx / 600, 32 / 600; gap = 0)
end
rowgap!(strips, 2)
buoyancy_colorbar(fig[1:2, 3]; width = 6, height = Relative(0.6))
wall_panels!(fig[3, 1:3], "gh")
rowgap!(fig.layout, 10)
resize_to_layout!(fig)

save("./Output/single_H100_setup.pdf", fig, pt_per_unit = 1)
save("./Output/single_H100_setup.png", fig, px_per_unit = 6)
#%%
