using CairoMakie
using Makie

include("../benchmarks/rough_rayleigh_benard_setup.jl")

# Lengths are in units of the domain height; Δ is the grid spacing, 1/448 on A100 and 1/600 on H100.
triangle(η, Δ) = 8Δ - abs(mod(η, 16Δ) - 8Δ)
pyramid(x, y, Δ) = min(triangle(x, Δ), triangle(y, Δ))
buoyancy(x, z, Lx) = -z + 1/2 + tanh((x - Lx / 2) / (Lx / 20)) / 4

colormap = :balance
colorrange = (-0.75, 0.75)
azimuth = 1.45π

"Square pyramids on the bottom wall of `[x₀, x₁] × [0, Ly]`, their faces shaded by the way they face."
function pyramids(x₀, x₁, Ly, Δ)
    h = 8Δ
    points = Point3f[]
    colors = RGBf[]
    shades = RGBf.(Makie.to_color.([:gray75, :gray55, :gray40, :gray90]))
    for a in x₀:2h:x₁-h, c in 0:2h:Ly-h
        apex = Point3f(a + h, c + h, h)
        corners = Point3f.([(a, c, 0), (a + 2h, c, 0), (a + 2h, c + 2h, 0), (a, c + 2h, 0)])
        for k in 1:4
            append!(points, (corners[k], corners[mod1(k + 1, 4)], apex))
            append!(colors, fill(shades[k], 3))
        end
    end
    return points, reshape(1:length(points), 3, :)', colors
end

"""
    block!(ax, x₀, x₁, Lx, Ly, Δ; shift)

One GPU's block `[x₀, x₁] × [0, Ly] × [0, 1]` of a domain of length `Lx`, drawn `shift` further along x:
pyramids on the bottom wall, and the initial buoyancy with contours on the back wall and the faces at `x₀` and `x₁`.
"""
function block!(ax, x₀, x₁, Lx, Ly, Δ; shift)
    x = range(x₀, x₁, length = max(2, round(Int, 200 * (x₁ - x₀))))
    z = range(0, 1, length = 100)
    b = buoyancy.(x, z', Lx)
    x_face!(x) = surface!(ax, fill(x + shift, 2, length(z)), [0, Ly] .* ones(1, length(z)), repeat(z', 2);
                          color = repeat(buoyancy.(x, z', Lx), 2), colormap, colorrange, shading = NoShading, rasterize = 5)

    # Far to near for a camera on the -x, -y side
    surface!(ax, x .+ shift, fill(Ly, length(x), length(z)), repeat(z', length(x));
             color = b, colormap, colorrange, shading = NoShading, rasterize = 5)
    contour!(ax, x .+ shift, z, b; levels = -0.625:0.125:0.625, color = :black, linewidth = 0.4, transformation = (:xz, Ly))
    x_face!(x₁)
    points, faces, colors = pyramids(x₀, x₁, Ly, Δ)
    mesh!(ax, [p + Point3f(shift, 0, 0) for p in points], faces; color = colors, shading = NoShading, rasterize = 5)
    x_face!(x₀)

    corners = [(x₀, 0, 0), (x₁, 0, 0), (x₁, Ly, 0), (x₀, Ly, 0)]
    edges = [Point3f.([c, (c[1], c[2], 1)]) for c in corners]
    for z in (0, 1)
        push!(edges, Point3f.([(c[1], c[2], z) for c in [corners; corners[1:1]]]))
    end
    for e in edges
        lines!(ax, [p + Point3f(shift, 0, 0) for p in e]; color = :black, linewidth = 0.5)
    end
    return nothing
end

"""
    domain!(ax, ngpus, Lx, Ly, Δ; gap, xticks)

A domain of size `Lx × Ly × 1` split into `ngpus` blocks along x, each pulled `gap` apart. `xticks` are in
unshifted x.
"""
function domain!(ax, ngpus, Lx, Ly, Δ; gap, xticks)
    w = Lx / ngpus
    shift(x) = gap * min(floor(Int, x / w), ngpus - 1)
    for r in ngpus:-1:1
        block!(ax, (r - 1) * w, r * w, Lx, Ly, Δ; shift = (r - 1) * gap)
    end

    ax.xticks = ([x + shift(x) for x in xticks], string.(xticks))
    limits!(ax, 0, Lx + (ngpus - 1) * gap, 0, Ly, 0, 1)
    ax.aspect = (Lx + (ngpus - 1) * gap, Ly, 1)
    return nothing
end

function axis3(position, title)
    return Axis3(position; title, azimuth, elevation = 0.2π, viewmode = :fitzoom, titlegap = 0, titlesize = 9,
                 xlabel = "x", ylabel = "y", zlabel = "z", zticks = [0, 1], yticksvisible = false, yticklabelsvisible = false, xgridvisible = false, ygridvisible = false,
                 zgridvisible = false, xlabeloffset = 15, ylabeloffset = 5, zlabeloffset = 15, protrusions = (30, 0, 0, 12))
end

"The immersed bottom wall under 2 × 2 pyramids as columns of solid cells, in units of grid spacings."
function stepped_pyramids!(ax)
    n = 32
    H = [count(k -> k - 0.5 <= pyramid(i - 0.5, j - 0.5, 1), 1:8) for i in 1:n, j in 1:n]
    shades = RGBf.(Makie.to_color.([:gray88, :gray68, :gray52])) # top, -y and -x faces
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
    ax3 = axis3(layout[1, 1], "($(letters[1])) Immersed bottom wall")
    ax3.elevation = 0.15π
    hidedecorations!(ax3)
    hidespines!(ax3)
    stepped_pyramids!(ax3)

    ax = Axis(layout[1, 2]; aspect = DataAspect(), xlabel = "x / Δx", ylabel = "z / Δz", xticks = 0:8:32, yticks = 0:4:8,
              title = "($(letters[2])) Solid cells through the apexes", titlesize = 9, titlegap = 2)
    cells!(ax)
    colsize!(layout, 1, Auto(1.4))
    return nothing
end

buoyancy_colorbar(position) = Colorbar(position; colormap, limits = colorrange, label = "Initial buoyancy b",
                                       ticks = -0.5:0.5:0.5, width = 6)

#%% Weak scaling on A100: one 448³ block per GPU stacked in x
fig = Figure(size = (510, 420), fontsize = 9, figure_padding = (2, 8, 2, 4))

domain!(axis3(fig[1, 1], "(a) 1 GPU, 448³ points"), 1, 1, 1, 1 / 448; gap = 0, xticks = [0, 1])
domain!(axis3(fig[1, 2], "(b) 4 GPUs, 448³ points each"), 4, 4, 1, 1 / 448; gap = 0.25, xticks = 0:4)
domain!(axis3(fig[2, 1:2], "(c) 16 GPUs, 448³ points each"), 16, 16, 1, 1 / 448; gap = 0.25, xticks = 0:4:16)
pyramid_panels!(fig[3, 1:2], "de")
buoyancy_colorbar(fig[1:2, 3])
colsize!(fig.layout, 1, Auto(1))
colsize!(fig.layout, 2, Auto(3))

save("./Output/weakscaling_setup.pdf", fig, pt_per_unit = 1)
save("./Output/weakscaling_setup.png", fig, px_per_unit = 6)

#%% Strong scaling on A100: the 448³ block split in x
fig = Figure(size = (510, 330), fontsize = 9, figure_padding = (2, 8, 2, 4))

for (col, (n, gap)) in enumerate(zip((1, 4, 16), (0, 0.12, 0.04)))
    points = n == 1 ? "448³ points" : "$(448 ÷ n) × 448² each"
    domain!(axis3(fig[1, col], "($('a' + col - 1)) $n GPU$(n > 1 ? "s" : ""), $points"), n, 1, 1, 1 / 448;
            gap, xticks = [0, 0.5, 1])
end
pyramid_panels!(fig[2, 1:3], "de")
buoyancy_colorbar(fig[1, 4])
rowsize!(fig.layout, 2, Relative(0.4))

save("./Output/strongscaling_setup.pdf", fig, pt_per_unit = 1)
save("./Output/strongscaling_setup.png", fig, px_per_unit = 6)

#%% One H100: the largest square, 600 × 600 × 600, and strip, 11250 × 32 × 600, of the isotropic sweep
fig = Figure(size = (510, 400), fontsize = 9, figure_padding = (2, 8, 2, 4))

square = axis3(fig[1, 1], "(a) Square, Nx × Nx × 600")
domain!(square, 1, 1, 1, 1 / 600; gap = 0, xticks = [0, 1])
for Nx in (64, 128, 256, 512)
    L = Nx / 600
    lines!(square, Point3f.([(0, L, 0.03), (L, L, 0.03), (L, 0, 0.03)]); color = :black, linewidth = 0.6, linestyle = :dash)
end

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

strip = axis3(fig[2, 1:2], "(c) Strip, Nx × 32 × 600")
domain!(strip, 1, 11250 / 600, 32 / 600, 1 / 600; gap = 0, xticks = 0:5:15)
for Nx in (128, 512, 2048, 8192)
    L = Nx / 600
    lines!(strip, Point3f.([(L, 0, 0), (L, 32 / 600, 0), (L, 32 / 600, 1)]); color = :black, linewidth = 0.6, linestyle = :dash)
end
pyramid_panels!(fig[3, 1:2], "de")
buoyancy_colorbar(fig[1:2, 3])

save("./Output/single_H100_setup.pdf", fig, pt_per_unit = 1)
save("./Output/single_H100_setup.png", fig, px_per_unit = 6)
#%%
