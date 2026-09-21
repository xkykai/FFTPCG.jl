using Oceananigans
using Oceananigans: prognostic_fields
using Oceananigans.Architectures: architecture
using Oceananigans.OutputWriters: with_architecture_suffix, write_output!
using Oceananigans.Operators: Δzᶜᶜᶜ
using Oceananigans.BoundaryConditions: NoFluxBoundaryCondition
using Oceananigans.Models.NonhydrostaticModels: ConjugateGradientPoissonSolver, nonhydrostatic_pressure_solver
using Oceananigans.Solvers: DiagonallyDominantPreconditioner, ColumnwiseTridiagonalPreconditioner
using Oceananigans.Grids: with_number_type
using Random

include("../utils/construct_stretched_spacing.jl")

const Ra = 1e6
const ν = κ = 1 / sqrt(Ra)
const spinup_time = 20

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

    return ConjugateGradientPoissonSolver(grid, maxiter=10000; preconditioner)
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
    setup_grid(arch, N, grid_type; Lx=1)

Pyramid-roughened Rayleigh-Bénard grid with `N` points per unit length in x and y, `Lx` unit
boxes in x, and `N` (`isotropic`) or `8N` (`anisotropic`, `stretched`) points in z.
"""
function setup_grid(arch, N, grid_type; Lx = 1)
    Ly = Lz = 1
    Nz = grid_type == "isotropic" ? N : 8N
    z = grid_type == "stretched" ? stretched_z_faces(Nz, Lz) : (0, Lz)

    grid = RectilinearGrid(arch, Float64,
                           size = (N * Lx, N, Nz),
                           halo = (6, 6, 6),
                           x = (0, Lx),
                           y = (0, Ly),
                           z = z,
                           topology = (Bounded, Bounded, Bounded))

    Nr = 8 # roughness elements per unit length
    h = 1 / (2Nr)
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

# Stable on the finest benchmarked grid: `stretched` with N = 256
const Δt = let Δz = minimum(diff(stretched_z_faces(8 * 256, 1)))
    min(Δz, Δz^2 / max(ν, κ)) / 3
end

@inline rayleigh_benard_buoyancy(x, y, z, t) = ifelse(z > 1 / 2, -1/2, 1/2)

function rough_rayleigh_benard_boundary_conditions(::ExplicitTimeDiscretization)
    no_slip_bc = ValueBoundaryCondition(0)

    u_bcs = FieldBoundaryConditions(no_slip_bc)
    v_bcs = FieldBoundaryConditions(no_slip_bc)
    w_bcs = FieldBoundaryConditions(no_slip_bc)
    b_bcs = FieldBoundaryConditions(top=ValueBoundaryCondition(-1/2), bottom=ValueBoundaryCondition(1/2),
                                    immersed=ValueBoundaryCondition(rayleigh_benard_buoyancy))

    return (u=u_bcs, v=v_bcs, w=w_bcs, b=b_bcs)
end

# Diffusive flux J = Fₑ + λφ from a wall held at `p.φ` half a cell from the adjacent cell center.
# J is upward at the domain walls and points into the fluid at immersed facets.
@inline wall_conductance(i, j, k, grid, p) = 2 * p.K / Δzᶜᶜᶜ(i, j, k, grid)

@inline bottom_wall_flux(i, j, grid, clock, fields, p)             =  wall_conductance(i, j, 1, grid, p) * p.φ
@inline bottom_wall_coefficient(i, j, grid, clock, fields, p)      = -wall_conductance(i, j, 1, grid, p)
@inline top_wall_flux(i, j, grid, clock, fields, p)                = -wall_conductance(i, j, size(grid, 3), grid, p) * p.φ
@inline top_wall_coefficient(i, j, grid, clock, fields, p)         =  wall_conductance(i, j, size(grid, 3), grid, p)
@inline immersed_wall_flux(i, j, k, grid, clock, fields, p)        =  wall_conductance(i, j, k, grid, p) * p.φ
@inline immersed_wall_coefficient(i, j, k, grid, clock, fields, p) = -wall_conductance(i, j, k, grid, p)

function implicit_wall_conditions(default_bc; K, bottom, top, sides)
    wall(flux, coefficient, φ) = IMEXFluxBoundaryCondition(flux, coefficient; discrete_form = true, parameters = (; φ, K))

    immersed = ImmersedBoundaryCondition(west = sides, east = sides, south = sides, north = sides,
                                         bottom = wall(immersed_wall_flux, immersed_wall_coefficient, bottom),
                                         top = wall(immersed_wall_flux, immersed_wall_coefficient, top))

    return FieldBoundaryConditions(default_bc; immersed,
                                   bottom = wall(bottom_wall_flux, bottom_wall_coefficient, bottom),
                                   top = wall(top_wall_flux, top_wall_coefficient, top))
end

function rough_rayleigh_benard_boundary_conditions(::VerticallyImplicitTimeDiscretization)
    no_slip_bc = ValueBoundaryCondition(0)

    u_bcs = implicit_wall_conditions(no_slip_bc; K = ν, bottom = 0, top = 0, sides = no_slip_bc)
    v_bcs = implicit_wall_conditions(no_slip_bc; K = ν, bottom = 0, top = 0, sides = no_slip_bc)
    w_bcs = FieldBoundaryConditions(no_slip_bc)
    b_bcs = implicit_wall_conditions(NoFluxBoundaryCondition(); K = κ, bottom = 1/2, top = -1/2,
                                     sides = ValueBoundaryCondition(rayleigh_benard_buoyancy))

    return (u=u_bcs, v=v_bcs, w=w_bcs, b=b_bcs)
end

function setup_model(grid, pressure_solver; seed = 1234, time_discretization = ExplicitTimeDiscretization())
    model = NonhydrostaticModel(grid; pressure_solver,
                                advection = WENO(order=9),
                                closure = ScalarDiffusivity(time_discretization; ν, κ),
                                tracers = :b,
                                buoyancy = BuoyancyTracer(),
                                boundary_conditions = rough_rayleigh_benard_boundary_conditions(time_discretization))

    Random.seed!(seed)
    bᵢ(x, y, z) = rand() * 1e-2 - z + 0.5
    set!(model, b=bᵢ)

    return model
end

spun_up_filepath(grid, dir) = joinpath(dir, with_architecture_suffix(architecture(grid), "spun_up.jld2", ".jld2"))

"""
    spin_up!(grid, dir; seed = 1234)

Run the model with the FFT pressure solver and vertically implicit diffusion to `spinup_time`
and save its prognostic fields in `dir`, unless they are already there.
"""
function spin_up!(grid, dir; seed = 1234)
    isfile(spun_up_filepath(grid, dir)) && return nothing

    model = setup_model(grid, build_solver(grid, "FFT"); seed, time_discretization = VerticallyImplicitTimeDiscretization())
    simulation = Simulation(model; Δt, stop_time = spinup_time)
    conjure_time_step_wizard!(simulation; cfl = 0.5, max_Δt = minimum_xspacing(grid)^2 / max(ν, κ) / 3)
    add_callback!(simulation, sim -> @info("Spin-up reached t = $(time(sim))"), TimeInterval(1))
    run!(simulation)

    time(simulation) ≥ spinup_time || error("Spin-up stopped at t = $(time(simulation)) before reaching t = $spinup_time")

    writer = JLD2Writer(model, prognostic_fields(model); dir, filename = "spun_up.jld2", schedule = IterationInterval(1),
                        array_type = Array{Float64})
    write_output!(writer, model)

    return nothing
end

"""
    set_spun_up_state!(model, dir)

Set the prognostic fields of `model` to those saved by `spin_up!` in `dir`.
"""
function set_spun_up_state!(model, dir)
    filepath = spun_up_filepath(model.grid, dir)
    fields = (name => FieldTimeSeries(filepath, string(name); grid = model.grid)[end] for name in keys(prognostic_fields(model)))
    set!(model; fields...)
    return nothing
end
