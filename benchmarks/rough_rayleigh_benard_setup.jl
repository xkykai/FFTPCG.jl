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

    return ConjugateGradientPoissonSolver(grid; preconditioner)
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
spacing is `1/Nz` for `isotropic` grids and `4/Nz` for `anisotropic` and `stretched` grids. Every
pyramid is 16 horizontal points wide and half as tall, so larger grids hold more pyramids.
`stretched` grids cluster points in z near the top and bottom.
"""
function setup_grid(arch, grid_type, Nx, Ny, Nz)
    N = grid_type == "isotropic" ? Nz : Nz ÷ 4 # horizontal points per unit length
    Δx = 1 / N
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

    h = 8 / N # pyramid height and half-width
    ϵ = 1e-9 * h # cell centres on a pyramid face are solid despite rounding

    # Pyramids where triangle waves of period 2h in x and y cross
    @inline triangle(η) = h - abs(mod(η, 2h) - h)
    @inline pyramid(x, y) = min(triangle(x), triangle(y)) + ϵ
    @inline mask(x, y, z) = (z <= pyramid(x, y)) | (z >= Lz - pyramid(x, y))

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
