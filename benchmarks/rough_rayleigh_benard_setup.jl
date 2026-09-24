using Oceananigans
using Oceananigans.Models.NonhydrostaticModels: ConjugateGradientPoissonSolver, nonhydrostatic_pressure_solver
using Oceananigans.Solvers: DiagonallyDominantPreconditioner, ColumnwiseTridiagonalPreconditioner
using Oceananigans.Grids: with_number_type
using Random

include("../utils/construct_stretched_spacing.jl")

const Ra = 1e8
const ν = κ = 1 / sqrt(Ra)

const GRID_TYPES = ("isotropic", "anisotropic", "stretched")
const PRECONDITIONERS = ("FFT", "no", "FFT64", "FFT32", "DiagonallyDominant", "ColumnwiseTridiagonal")

output_suffix(grid_type) = grid_type == "isotropic" ? "" : "_$grid_type"

@inline function local_roughness_bottom(η, η₀, h)
    if η > η₀ - h && η <= η₀
        return η + h - η₀
    elseif η > η₀ && η <= η₀ + h
        return -η + h + η₀
    else
        return 0
    end
end

@inline function local_roughness_top(η, η₀, h)
    if η > η₀ - h && η <= η₀
        return -η - h + η₀
    elseif η > η₀ && η <= η₀ + h
        return η - h - η₀
    else
        return 0
    end
end

function build_solver(grid, precond_name)
    precond_name == "FFT" && return nothing

    if precond_name == "no"
        preconditioner = nothing
    elseif precond_name == "FFT64"
        preconditioner = nonhydrostatic_pressure_solver(grid.underlying_grid, nothing)
    elseif precond_name == "FFT32"
        reduced_precision_grid = with_number_type(Float32, grid.underlying_grid)
        preconditioner = nonhydrostatic_pressure_solver(reduced_precision_grid, nothing)
    elseif precond_name == "DiagonallyDominant"
        preconditioner = DiagonallyDominantPreconditioner()
    elseif precond_name == "ColumnwiseTridiagonal"
        preconditioner = ColumnwiseTridiagonalPreconditioner(grid)
    end

    return ConjugateGradientPoissonSolver(grid, maxiter=20000; preconditioner)
end

function stretched_z_faces(Nz, Lz)
    a = 15
    b = 15
    c = 1 / (2Nz)
    d = 1 / (Nz / 2)
    f = 0.15
    g = 0.85

    spacing(z) = stretched_tanh(z, a, b, c, d, f, g)

    return stretched_grid_from_spacing(spacing, 0, Lz, Nz + 1)
end

"""
    setup_grid(arch, grid_type, Nx, Ny, Nz)

Pyramid-roughened Rayleigh-Bénard grid of unit height with `Nx × Ny × Nz` points. The horizontal
spacing is 1/512 (`isotropic`) or 1/256 (`anisotropic`, `stretched`), and every pyramid is 1/32
wide and 1/64 tall, so larger grids hold more pyramids. `stretched` grids cluster points in z near
the top and bottom.
"""
function setup_grid(arch, grid_type, Nx, Ny, Nz)
    Δx = grid_type == "isotropic" ? 1/512 : 1/256
    Lx = Nx * Δx
    Ly = Ny * Δx
    Lz = 1
    z = grid_type == "stretched" ? stretched_z_faces(Nz, Lz) : (0, Lz)

    grid = RectilinearGrid(arch, Float64,
                           size = (Nx, Ny, Nz),
                           halo = (6, 6, 6),
                           x = (0, Lx),
                           y = (0, Ly),
                           z = z,
                           topology = (Bounded, Bounded, Bounded))

    h = 1/64 # pyramid height and half-width
    x₀s = h:2h:Lx-h
    y₀s = h:2h:Ly-h

    # Woven pattern: ridges in x and y directions create pyramids
    @inline function roughness_bottom(x, y, z)
        z_rough_x = sum([local_roughness_bottom(x, x₀, h) for x₀ in x₀s])
        z_rough_y = sum([local_roughness_bottom(y, y₀, h) for y₀ in y₀s])
        z_rough = min(z_rough_x, z_rough_y)
        return z <= z_rough
    end

    @inline function roughness_top(x, y, z)
        z_rough_x = sum([local_roughness_top(x, x₀, h) for x₀ in x₀s])
        z_rough_y = sum([local_roughness_top(y, y₀, h) for y₀ in y₀s])
        z_rough = max(z_rough_x, z_rough_y)
        return z >= z_rough + Lz
    end

    @inline mask(x, y, z) = roughness_bottom(x, y, z) | roughness_top(x, y, z)

    return ImmersedBoundaryGrid(grid, GridFittedBoundary(mask))
end

function stable_timestep(grid)
    Δz = minimum_zspacing(grid)
    return min(Δz, Δz^2 / max(ν, κ)) / 3
end

function setup_model(grid, pressure_solver; seed = 1234)
    closure = ScalarDiffusivity(ν=ν, κ=κ)

    @inline function rayleigh_benard_buoyancy(x, y, z, t)
        above_centerline = z > 1 / 2
        return ifelse(above_centerline, -1/2, 1/2)
    end

    no_slip_bc = ValueBoundaryCondition(0)

    u_bcs = FieldBoundaryConditions(no_slip_bc)
    v_bcs = FieldBoundaryConditions(no_slip_bc)
    w_bcs = FieldBoundaryConditions(no_slip_bc)
    b_bcs = FieldBoundaryConditions(top=ValueBoundaryCondition(-1/2), bottom=ValueBoundaryCondition(1/2),
                                    immersed=ValueBoundaryCondition(rayleigh_benard_buoyancy))

    model = NonhydrostaticModel(grid; pressure_solver,
                                advection = WENO(order=9),
                                closure = closure,
                                tracers = :b,
                                buoyancy = BuoyancyTracer(),
                                boundary_conditions = (u=u_bcs, v=v_bcs, w=w_bcs, b=b_bcs))

    Random.seed!(seed)
    bᵢ(x, y, z) = rand() * 1e-2 - z + 0.5
    set!(model, b=bᵢ)

    return model
end
