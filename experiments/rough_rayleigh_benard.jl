using Oceananigans
using Printf
using JLD2
using Oceananigans.Models.NonhydrostaticModels: ConjugateGradientPoissonSolver, FFTBasedPoissonSolver
using Oceananigans.Models.NonhydrostaticModels: nonhydrostatic_pressure_solver
using Oceananigans.Models: buoyancy_field
using Oceananigans.Grids: with_number_type, minimum_xspacing, minimum_zspacing
using Oceananigans.Utils: launch!
using Oceananigans.Architectures: architecture
using Oceananigans.Operators
using KernelAbstractions: @kernel, @index
using Statistics
using CUDA
using ArgParse
using Glob

function parse_commandline()
    s = ArgParseSettings()
  
    @add_arg_table! s begin
      "--solver"
        help = "Type of pressure solver to use (FFT or CG)"
        arg_type = String
        default = "CG"
    end
    return parse_args(s)
end

include("setup_rough_rayleigh_benard_model.jl")

args = parse_commandline()
solver_type = args["solver"]

arch = GPU()

const Ra = 1e8
const Nr = 8
const g = 1
const α = 1
const β = 4
const Pr = 1
const ΔT = -1
const ΔS = 1
const H = 1
const Δb = (-α*ΔT + β*ΔS) * g

const ν = sqrt(Δb * g * H^3 * Pr / Ra)
const κ = ν / Pr
const Lx = 1
const Lz = 1
const Nx = 1024
const Nz = 512

grid = setup_stretched_grid(; Nr, arch, Nx, Nz)

if solver_type == "FFT"
    pressure_solver = nothing
    pressure_solver_str = solver_type
elseif solver_type == "CG"
    preconditioner = nonhydrostatic_pressure_solver(with_number_type(Float32, grid.underlying_grid), nothing)
    pressure_solver = ConjugateGradientPoissonSolver(grid, maxiter=80; preconditioner)
    pressure_solver_str = solver_type
end

model = setup_model(grid, pressure_solver; Ra)

filename = "rough_RB_seaiceformation_noslip_periodic_halfdomain_stretched_Ra_$(Ra)_Pr_$(Pr)_Nr_$(Nr)_Lx_$(Lx)_Lz_$(Lz)_Nx_$(Nx)_Nz_$(Nz)_$(pressure_solver_str)"

FILE_DIR = "./Data/$(filename)"
mkpath(FILE_DIR)

# Boundary T/S values (used below for the Nusselt-number diagnostic)
const T_top = 0
const T_bottom = 1

const S_top = 1
const S_bottom = 0

stop_time = 1001
Δx_min = minimum_xspacing(grid)
Δz_min = minimum_zspacing(grid)
advective_Δt = Δz_min / Δb
diffusive_Δt = min(Δx_min^2, Δz_min^2) / max(ν, κ)
Δt = min(advective_Δt, diffusive_Δt) / 10

simulation = Simulation(model; Δt, stop_time)
time_wizard = TimeStepWizard(cfl=0.6, max_change=1.05, max_Δt=diffusive_Δt / 5)

simulation.callbacks[:wizard] = Callback(time_wizard, IterationInterval(1))

u, v, w = model.velocities
T, S, c = model.tracers.T, model.tracers.S, model.tracers.c
b = buoyancy_field(model)

d = CenterField(grid)

@kernel function _divergence!(target_field, u, v, w, grid)
    i, j, k = @index(Global, NTuple)
    @inbounds target_field[i, j, k] = divᶜᶜᶜ(i, j, k, grid, u, v, w)
end

function compute_flow_divergence!(target_field, model)
    grid = model.grid
    u, v, w = model.velocities
    arch = architecture(grid)
    launch!(arch, grid, :xyz, _divergence!, target_field, u, v, w, grid)
    return nothing
end

function progress(sim)
    if pressure_solver isa ConjugateGradientPoissonSolver
        pressure_iters = iteration(pressure_solver)
    else
        pressure_iters = 0
    end

    msg = @sprintf("Iter: %d, time: %s, Δt: %s, Poisson iters: %d",
                    iteration(sim), prettytime(sim), prettytime(sim.Δt), pressure_iters)

    compute_flow_divergence!(d, sim.model)

    msg *= @sprintf(", max u: %6.3e, max w: %6.3e, max T: %6.3e, max S: %6.3e, max d: %6.3e, max pressure: %6.3e, mean pressure: %6.3e",
                    maximum(abs, sim.model.velocities.u),
                    maximum(abs, sim.model.velocities.w),
                    maximum(abs, sim.model.tracers.T),
                    maximum(abs, sim.model.tracers.S),
                    maximum(abs, d),
                    maximum(abs, sim.model.pressures.pNHS),
                    mean(sim.model.pressures.pNHS),
    )

    @info msg

    return nothing
end

simulation.callbacks[:progress] = Callback(progress, IterationInterval(100))

wb = Field(w * b)
∂b∂z = α * (T_top - T_bottom) / Lz - β * (S_top - S_bottom) / Lz
Nu = Average(1 - wb / (κ * ∂b∂z), dims=(1, 2))

Tbar = Average(T, dims=(1, 2))
Sbar = Average(S, dims=(1, 2))
bbar = Average(b, dims=(1, 2))
KEbar = Average(0.5 * (u^2 + w^2), dims=(1, 2, 3))

p = model.pressures.pNHS + model.pressures.pHY′

simulation.output_writers[:jld2] = JLD2Writer(model, (; u, w, T, S, c, b, d, p);
                                              filename = joinpath(FILE_DIR, "instantaneous_fields.jld2"),
                                              schedule = TimeInterval(50),
                                              with_halos = true)

simulation.output_writers[:averaged] = JLD2Writer(model, (; T = Tbar, S = Sbar, b = bbar, Nu);
                                              filename = joinpath(FILE_DIR, "averaged_fields.jld2"),
                                              schedule = AveragedTimeInterval(1000, window=500),
                                              with_halos = true)

simulation.output_writers[:KE] = JLD2Writer(model, (; KE = KEbar);
                                              filename = joinpath(FILE_DIR, "KE_fields.jld2"),
                                              schedule = AveragedTimeInterval(50, window=50),
                                              with_halos = true)

simulation.output_writers[:checkpoint] = Checkpointer(model;
                                                      dir = FILE_DIR,
                                                      schedule = TimeInterval(50))

checkpoint_files = glob("checkpoint*.jld2", FILE_DIR)
if !isempty(checkpoint_files)
    @info "Found checkpoint files, resuming from checkpoint"
    run!(simulation, pickup=true)
else
    @info "No checkpoint files found, starting fresh simulation"
    run!(simulation)
end

checkpoint_files = glob("checkpoint*.jld2", FILE_DIR)
if !isempty(checkpoint_files)
    @info "Simulation completed successfully, removing $(length(checkpoint_files)) checkpoint file(s)"
    for f in checkpoint_files
        rm(f)
    end
end
