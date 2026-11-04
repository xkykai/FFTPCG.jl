using Oceananigans
using Oceananigans.Models.NonhydrostaticModels: ConjugateGradientPoissonSolver, FFTBasedPoissonSolver
using Oceananigans.Grids: with_number_type
# using CairoMakie

include("../utils/construct_stretched_spacing.jl")

#####
##### Roughness topography
#####

@inline function local_roughness_top(η, η₀, half_width, h_element)
    if η > η₀ - half_width && η <= η₀
        return h_element / half_width * (η₀ - half_width - η)
    elseif η > η₀ && η <= η₀ + half_width
        return h_element / half_width * (η - η₀ - half_width)
    else
        return 0
    end
end

#####
##### Grid setup
#####

function setup_stretched_grid(; Nr, arch = GPU(), Nx = 512, Nz = 256)
    Lx = 1
    Lz = 1

    # Spacing is d/c = 4 times coarser for f < z < g than in the bottom
    # boundary layer and the roughness region
    a = 15
    b = 15
    c = 1 / (2Nz)
    d = 1 / (Nz/2)
    f = 0.15
    g = 0.35

    h(z) = stretched_tanh(z, a, b, c, d, f, g)

    # Face nodes for the Bounded z-direction (needs Nz + 1 points)
    zs = stretched_grid_from_spacing(h, 0, Lz, Nz + 1)

    # Periodic-in-x grid
    grid = RectilinearGrid(arch, Float64,
                           size = (Nx, Nz),
                           halo = (6, 6),
                           x = (0, Lx),
                           z = zs,
                           topology = (Periodic, Flat, Bounded))

    # Roughness elements along the top boundary
    hx = Lx / Nr / 2
    h_element = Lx / 2

    @inline function roughness_top(x, z)
        z_rough_x = zero(x)
        for n in 1:Nr
            x₀ = (2n - 1) * hx
            z_rough_x += local_roughness_top(x, x₀, hx, h_element)
        end
        return z >= z_rough_x + Lz
    end

    @inline mask(x, z) = roughness_top(x, z)

    return ImmersedBoundaryGrid(grid, GridFittedBoundary(mask))
end
    
#%%
#####
##### Model setup
#####
#
# `pressure_solver` is a solver, a function `grid -> solver`, or `nothing`
# for the Oceananigans default.

function setup_model(grid, pressure_solver; Ra)
    # Physical parameters
    g  = 1
    α  = 1
    β  = 4
    Pr = 1
    ΔT = -1
    ΔS = 1
    H  = 1
    Δb = (-α*ΔT + β*ΔS) * g

    ν = sqrt(Δb * g * H^3 * Pr / Ra)
    κ = ν / Pr

    Lz = 1

    closure = ScalarDiffusivity(ν=ν, κ=κ)

    equation_of_state = LinearEquationOfState(thermal_expansion=α, haline_contraction=β)
    buoyancy = SeawaterBuoyancy(; gravitational_acceleration=g, equation_of_state)

    # Build the pressure solver from the grid (accept a builder or a ready solver)
    solver = pressure_solver isa Function ? pressure_solver(grid) : pressure_solver

    # Boundary conditions
    T_top = 0
    T_bottom = 1
    S_top = 1
    S_bottom = 0

    @inline function rayleigh_benard_T(x, z, t)
        above_centerline = z > 1 / 2
        return ifelse(above_centerline, T_top, T_bottom)
    end

    @inline function rayleigh_benard_S(x, z, t)
        above_centerline = z > 1 / 2
        return ifelse(above_centerline, S_top, S_bottom)
    end

    no_slip_bc = ValueBoundaryCondition(0)

    u_bcs = FieldBoundaryConditions(top=no_slip_bc, bottom=no_slip_bc, immersed=no_slip_bc)
    v_bcs = FieldBoundaryConditions(top=no_slip_bc, bottom=no_slip_bc, immersed=no_slip_bc)
    w_bcs = FieldBoundaryConditions(immersed=no_slip_bc)

    T_bcs = FieldBoundaryConditions(top=ValueBoundaryCondition(T_top), bottom=ValueBoundaryCondition(T_bottom),
                                    immersed=ValueBoundaryCondition(rayleigh_benard_T))
    S_bcs = FieldBoundaryConditions(top=ValueBoundaryCondition(S_top), bottom=ValueBoundaryCondition(S_bottom),
                                    immersed=ValueBoundaryCondition(rayleigh_benard_S))

    boundary_conditions = (u=u_bcs, v=v_bcs, w=w_bcs, T=T_bcs, S=S_bcs)

    model = NonhydrostaticModel(grid; pressure_solver = solver,
                                advection = Centered(),
                                closure,
                                tracers = (:T, :S, :c),
                                buoyancy,
                                boundary_conditions)

    # Initial conditions
    Tᵢ(x, z) = T_bottom + rand() * 1e-5
    Sᵢ(x, z) = S_bottom + rand() * 1e-5
    c₁(x, z) = 1

    set!(model, T=Tᵢ, c=c₁, S=Sᵢ)

    return model
end
