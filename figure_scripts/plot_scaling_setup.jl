using CairoMakie
using Makie

const Δ = 1 / 448 # grid spacing of the isotropic grid, 448³ points per A100
const h = 8Δ # pyramid height and half-width

triangle(η) = h - abs(mod(η, 2h) - h)
pyramid(x, y) = min(triangle(x), triangle(y))
buoyancy(x, z, Lx) = -z + 1/2 + tanh((x - Lx / 2) / (Lx / 20)) / 4

const ys = 7.5Δ # x-z slices pass through a row of pyramid apexes
solid(x, z) = (z <= pyramid(x, ys)) | (z >= 1 - pyramid(x, ys))

colormap = :balance
colorrange = (-0.75, 0.75)

"Square pyramids on the bottom wall of `[x₀, x₁] × [0, 1]`, their faces shaded by the way they face."
function pyramids(x₀, x₁)
    points = Point3f[]
    colors = RGBf[]
    shades = RGBf.(Makie.to_color.([:gray75, :gray55, :gray40, :gray90]))
    for a in x₀:2h:x₁-h, c in 0:2h:1-h
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
    block!(ax, x₀, x₁, Lx; shift)

One GPU's block `[x₀, x₁] × [0, 1] × [0, 1]` of a domain of length `Lx`, drawn `shift` further along x:
pyramids on the bottom wall and the initial buoyancy on the back and left walls.
"""
function block!(ax, x₀, x₁, Lx; shift = 0)
    x = range(x₀, x₁, length = max(2, round(Int, 200 * (x₁ - x₀))))
    z = range(0, 1, length = 100)
    b = buoyancy.(x, z', Lx)
    X = x .+ shift

    surface!(ax, X, fill(1, length(x), length(z)), repeat(z', length(x)); color = b, colormap, colorrange, shading = NoShading, rasterize = 5)
    surface!(ax, fill(x₀ + shift, 2, length(z)), [0, 1] .* ones(1, length(z)), repeat(z', 2);
             color = repeat(buoyancy.(x₀, z', Lx), 2), colormap, colorrange, shading = NoShading, rasterize = 5)

    points, faces, colors = pyramids(x₀, x₁)
    mesh!(ax, [p + Point3f(shift, 0, 0) for p in points], faces; color = colors, shading = NoShading, rasterize = 5)

    corners = [(x₀, 0, 0), (x₁, 0, 0), (x₁, 1, 0), (x₀, 1, 0)]
    edges = [Point3f.([c, (c[1], c[2], 1)]) for c in corners]
    for z in (0, 1)
        push!(edges, Point3f.([(c[1], c[2], z) for c in [corners; corners[1:1]]]))
    end
    for e in edges
        lines!(ax, [p + Point3f(shift, 0, 0) for p in e]; color = :black, linewidth = 0.5)
    end
    return nothing
end

"Domain of `ngpus` blocks along x, each pulled `gap` apart, for a domain of length `Lx`."
function exploded!(ax, ngpus, Lx; gap)
    w = Lx / ngpus
    for r in 1:ngpus
        block!(ax, (r - 1) * w, r * w, Lx; shift = (r - 1) * gap)
    end
    xlims!(ax, 0, Lx + (ngpus - 1) * gap)
    ylims!(ax, 0, 1)
    zlims!(ax, 0, 1)
    return nothing
end

"x-z slice of the initial buoyancy at `y = ys` with rank boundaries dashed."
function slice!(ax, ngpus, Lx)
    x = range(Δ / 2, Lx - Δ / 2, step = Δ)
    z = range(Δ / 2, 1 - Δ / 2, step = Δ)
    b = [solid(x, z) ? NaN : buoyancy(x, z, Lx) for x in x, z in z]
    walls = [solid(x, z) ? 1.0 : NaN for x in x, z in z]
    hm = heatmap!(ax, x, z, b; colormap, colorrange, rasterize = 5)
    heatmap!(ax, x, z, walls; colormap = [:gray40, :gray40], rasterize = 5)
    contour!(ax, x, z, b; levels = -0.625:0.125:0.625, color = :black, linewidth = 0.4)
    vlines!(ax, (1:ngpus-1) .* (Lx / ngpus); color = :black, linewidth = 0.8, linestyle = :dash)
    xlims!(ax, 0, Lx)
    ylims!(ax, 0, 1)
    return hm
end

"Grid cells on the bottom wall at the slice: each pyramid is 16 cells wide and 8 tall."
function cells!(ax)
    i = 0:31
    k = 0:9
    cells = [solid((i + 0.5) * Δ, (k + 0.5) * Δ) ? 1.0 : NaN for i in i, k in k]
    heatmap!(ax, i .+ 0.5, k .+ 0.5, cells; colormap = [:gray55, :gray55])
    vlines!(ax, 0:32; color = :gray80, linewidth = 0.3)
    hlines!(ax, 0:10; color = :gray80, linewidth = 0.3)
    lines!(ax, 0:32, [pyramid(i * Δ, ys) / Δ for i in 0:32]; color = :black, linewidth = 0.6)
    limits!(ax, 0, 32, 0, 10)
    return nothing
end

"The immersed bottom wall under 2 × 2 pyramids as columns of solid cells, in units of grid spacings."
function stepped_pyramids!(ax)
    n = 32
    H = [count(k -> (k - 0.5) * Δ <= pyramid((i - 0.5) * Δ, (j - 0.5) * Δ), 1:8) for i in 1:n, j in 1:n]
    shades = RGBf.(Makie.to_color.([:gray88, :gray68, :gray52])) # top, -y and +x faces
    faces = [1 2 3; 1 3 4]

    # Back to front for a camera on the +x, -y side
    for (i, j) in sort(vec(collect(Iterators.product(1:n, 1:n))), by = ((i, j),) -> i - 2.5j)
        z = H[i, j]
        x₀, x₁, y₀, y₁ = i - 1, i, j - 1, j
        top = Point3f.([(x₀, y₀, z), (x₁, y₀, z), (x₁, y₁, z), (x₀, y₁, z)])
        front = Point3f.([(x₀, y₀, 0), (x₁, y₀, 0), (x₁, y₀, z), (x₀, y₀, z)])
        side = Point3f.([(x₁, y₀, 0), (x₁, y₁, 0), (x₁, y₁, z), (x₁, y₀, z)])
        for (quad, shade) in zip((top, front, side), shades)
            mesh!(ax, quad, faces; color = shade, shading = NoShading)
        end
    end
    limits!(ax, 0, n, 0, n, 0, 8)
    return nothing
end

function axis3(position, title; elevation = 0.12π)
    ax = Axis3(position; title, elevation, aspect = :data, azimuth = 1.62π, protrusions = (0, 0, 0, 12),
               viewmode = :fitzoom, titlegap = 0, titlesize = 9)
    hidedecorations!(ax)
    hidespines!(ax)
    ax.xgridvisible = ax.ygridvisible = ax.zgridvisible = false
    return ax
end

"Grey lines from the front bottom corners `x₀` and `x₁` of the 3D domain in `ax3` to the top corners of the slice `ax`."
function connect!(fig, ax3, x₀, x₁, ax)
    Makie.update_state_before_display!(fig)
    from = [Point2f(Makie.project(ax3.scene, Point3f(x, 0, 0))) + minimum(viewport(ax3.scene)[]) for x in (x₀, x₁)]
    box = viewport(ax.scene)[]
    to = [Point2f(minimum(box)[1], maximum(box)[2]), Point2f(maximum(box)...)]
    for (p, q) in zip(from, to)
        lines!(fig.scene, [p, q]; space = :pixel, color = :gray50, linewidth = 0.5, linestyle = :dot)
    end
    return nothing
end

function pyramid_panels!(position, letters)
    layout = GridLayout(position)
    stepped_pyramids!(axis3(layout[1, 1], "($(letters[1])) Immersed bottom wall"; elevation = 0.15π))

    ax = Axis(layout[1, 2]; aspect = DataAspect(), xlabel = "x / Δx", ylabel = "z / Δz", xticks = 0:8:32, yticks = 0:4:8,
              title = "($(letters[2])) Solid cells on the slice", titlesize = 9, titlegap = 2)
    cells!(ax)
    colsize!(layout, 1, Auto(1.4))
    return nothing
end

slice_axis(position, letter, ngpus, Lx) =
    Axis(position; title = rich("($letter) $ngpus GPU$(ngpus > 1 ? "s" : ""), L", subscript("x"), " = $Lx"), aspect = DataAspect(),
         xlabel = "x", ylabel = "z", yticks = 0:0.5:1, titlesize = 9, titlegap = 2)

#%% Weak scaling: one block per GPU stacked in x
fig = Figure(size = (510, 470), fontsize = 9, figure_padding = (2, 8, 2, 4))

gap = 0.25
ax1 = axis3(fig[1, 1], "(a) 1 GPU, 448³ points")
exploded!(ax1, 1, 1; gap)
ax4 = axis3(fig[1, 2], "(b) 4 GPUs, 448³ points each")
exploded!(ax4, 4, 4; gap)

slices = [slice_axis(fig[2, 1], 'c', 1, 1), slice_axis(fig[2, 2], 'd', 4, 4), slice_axis(fig[3, 1:2], 'e', 16, 16)]
hm = [slice!(ax, n, n) for (ax, n) in zip(slices, (1, 4, 16))][1]
Colorbar(fig[2:3, 3], hm; label = "Initial buoyancy b", ticks = -0.5:0.5:0.5, width = 6)
pyramid_panels!(fig[4, 1:2], "fg")

colsize!(fig.layout, 1, Auto(1))
colsize!(fig.layout, 2, Auto(4))
rowsize!(fig.layout, 1, Aspect(1, 0.9))
rowsize!(fig.layout, 2, Aspect(1, 1))
rowsize!(fig.layout, 3, Aspect(1, 0.37))
rowsize!(fig.layout, 4, Aspect(1, 1.1))
resize_to_layout!(fig)
connect!(fig, ax1, 0, 1, slices[1])
connect!(fig, ax4, 0, 4 + 3gap, slices[2])

display(fig)
save("./Output/weakscaling_setup.pdf", fig, pt_per_unit = 1)
save("./Output/weakscaling_setup.png", fig, px_per_unit = 6)

#%% Strong scaling: the 1-GPU block split in x
fig = Figure(size = (510, 400), fontsize = 9, figure_padding = (2, 8, 2, 4))

ngpus = (1, 4, 16)
gaps = (0, 0.12, 0.04)
volumes = [axis3(fig[1, col], "($('a' + col - 1)) $n GPU$(n > 1 ? "s" : ""), " * (n == 1 ? "448³ points" : "$(448 ÷ n) × 448² each"))
           for (col, n) in enumerate(ngpus)]
slices = [slice_axis(fig[2, col], 'd' + col - 1, n, 1) for (col, n) in enumerate(ngpus)]
for (volume, slice, n, gap) in zip(volumes, slices, ngpus, gaps)
    exploded!(volume, n, 1; gap)
    global hm = slice!(slice, n, 1)
end
Colorbar(fig[2, 4], hm; label = "Initial buoyancy b", ticks = -0.5:0.5:0.5, width = 6)
pyramid_panels!(fig[3, 1:3], "gh")

rowsize!(fig.layout, 1, Aspect(1, 0.9))
rowsize!(fig.layout, 2, Aspect(1, 1))
rowsize!(fig.layout, 3, Aspect(1, 0.75))
resize_to_layout!(fig)
for (volume, slice, n, gap) in zip(volumes, slices, ngpus, gaps)
    connect!(fig, volume, 0, 1 + (n - 1) * gap, slice)
end

display(fig)
save("./Output/strongscaling_setup.pdf", fig, pt_per_unit = 1)
save("./Output/strongscaling_setup.png", fig, px_per_unit = 6)
#%%
