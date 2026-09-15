using Oceananigans
using Printf
using JLD2
using Oceananigans.Models.NonhydrostaticModels: ConjugateGradientPoissonSolver, FFTBasedPoissonSolver
using Oceananigans.Models.NonhydrostaticModels: nonhydrostatic_pressure_solver
using Oceananigans.Solvers: DiagonallyDominantPreconditioner, ColumnwiseTridiagonalPreconditioner
using Oceananigans.Grids: with_number_type
using Statistics
using CUDA
using BenchmarkTools
using Random

include("benchmark_utils.jl")

arch = GPU()

const Ra = 1e6
const ν = κ = 1 / sqrt(Ra)

#####
##### Model setup
#####

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

function setup_grid(N)
    Lx = Ly = Lz = 1

    grid = RectilinearGrid(arch, Float64,
                           size = (N, N, 8N), 
                           halo = (6, 6, 6),
                           x = (0, Lx),
                           y = (0, Ly),
                           z = (0, Lz),
                           topology = (Bounded, Bounded, Bounded))

    Nr = 8 # number of roughness elements
    hx = Lx / Nr / 2
    hy = Ly / Nr / 2
    x₀s = hx:2hx:Lx-hx
    y₀s = hy:2hy:Ly-hy

    # Woven pattern: ridges in x and y directions create pyramids
    @inline function roughness_bottom(x, y, z)
        # Ridges running in y-direction (triangular profile in x)
        z_rough_x = sum([local_roughness_bottom(x, x₀, hx) for x₀ in x₀s])
        # Ridges running in x-direction (triangular profile in y)
        z_rough_y = sum([local_roughness_bottom(y, y₀, hy) for y₀ in y₀s])
        # Take minimum to create pyramidal peaks
        z_rough = min(z_rough_x, z_rough_y)
        return z <= z_rough
    end

    @inline function roughness_top(x, y, z)
        z_rough_x = sum([local_roughness_top(x, x₀, hx) for x₀ in x₀s])
        z_rough_y = sum([local_roughness_top(y, y₀, hy) for y₀ in y₀s])
        # Take maximum for inverted pyramids
        z_rough = max(z_rough_x, z_rough_y)
        return z >= z_rough + Lz
    end

    @inline mask(x, y, z) = roughness_bottom(x, y, z) | roughness_top(x, y, z)

    grid = ImmersedBoundaryGrid(grid, GridFittedBoundary(mask))
    return grid
end

function initial_conditions!(model)
    Random.seed!(1234)
    bᵢ(x, y, z) = rand() * 1e-2 - z + 0.5

    set!(model, b=bᵢ)
end

function setup_model(grid, pressure_solver)
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

    initial_conditions!(model)
    return model
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

Ns = [16, 32, 64, 96, 128, 192, 256]
Δts = [min(1 / 8N, (1/(8N)^2) / max(ν, κ)) / 3 for N in Ns]

mkpath("./reports/")
filename = "single_H100_anisotropic.jld2"
FILE_PATH = joinpath("./reports/", filename)

# Check whether a key already exists in the output file (i.e. that
# combination has already been benchmarked and can be skipped).
function key_exists(file_path, key)
    isfile(file_path) || return false
    return jldopen(file_path, "r") do file
        haskey(file, key)
    end
end

warmup_nsteps = 50
nsteps = 50

preconditioners = ["FFT", "no", "FFT64", "FFT32", "DiagonallyDominant", "ColumnwiseTridiagonal"]

for (N, Δt) in zip(Ns, Δts), precond_name in preconditioners
    if key_exists(FILE_PATH, "$(N)/times/$(precond_name)")
        @info "Skipping $precond_name for N=$N (already benchmarked)"
        continue
    end
    @info "Benchmarking $precond_name for N=$N"

    grid = setup_grid(N)
    model = setup_model(grid, build_solver(grid, precond_name))

    results = benchmark_time_steps!(model, Δt, nsteps; warmup=warmup_nsteps)

    jldopen(FILE_PATH, "a") do file
        file["$(N)/times/$(precond_name)"] = results.stats
        file["$(N)/cg_iters/$(precond_name)"] = results.iterations
        file["$(N)/gpu_state/$(precond_name)"] = (initial = results.initial_state,
                                                  final = results.final_state,
                                                  elapsed = results.elapsed)
    end

    grid = nothing
    model = nothing
    GC.gc()
    CUDA.reclaim()
end
