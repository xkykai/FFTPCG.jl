using Oceananigans
using Oceananigans.Models.NonhydrostaticModels: ConjugateGradientPoissonSolver, FFTBasedPoissonSolver
using Oceananigans.Grids: with_number_type
using Interpolations
# using CairoMakie

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
##### Stretched grid generation
#####
"""
    stretched_tanh(x, a, b, c, d, f, g)

Return a smooth, localized profile based on the difference of two hyperbolic
tangent transitions.

Parameters:
- `x`: coordinate at which to evaluate the profile.
- `a`: steepness of the transition centered at `f`.
- `b`: steepness of the transition centered at `g`.
- `c`: baseline value outside the stretched interval.
- `d`: elevated value inside the stretched interval.
- `f`: center location of the first transition.
- `g`: center location of the second transition.
"""
function stretched_tanh(x, a, b, c, d, f, g)
    return c + (d - c) / 2 * (tanh(a * (x - f)) - tanh(b * (x - g)))
end

#%%
# xs = range(0, 1, length=100)
# hs = stretched_tanh.(xs, a, b, c, d, f, g)

# fig = Figure(size=(800, 200))
# ax = Axis(fig[1, 1], xlabel="x", ylabel="h(x)")
# lines!(ax, xs, hs)
# fig
#%%

function stretched_grid_from_spacing(h, a, b, N; M=10_000)
    # Fine auxiliary grid
    xfine = range(a, b, length=M)
    hfine = h.(xfine)

    # Grid density: larger where h is smaller
    ρ = 1 ./ hfine

    # Cumulative integral using trapezoidal rule
    s = zeros(M)
    for i in 2:M
        dx = xfine[i] - xfine[i-1]
        s[i] = s[i-1] + 0.5 * dx * (ρ[i] + ρ[i-1])
    end

    # Normalize cumulative coordinate to [0, 1]
    s ./= s[end]

    # Invert s(x): x(s)
    interp = linear_interpolation(s, collect(xfine), extrapolation_bc=Line())

    # Uniform points in computational space
    ξ = range(0, 1, length=N)

    return interp.(ξ)
end

#%%
# dx = diff(x)
# extrema(dx)
# #%%
# fig = Figure(size=(1600, 100))
# ax = Axis(fig[1, 1])
# scatter!(ax, x, zeros(length(x)))
# fig
# #%%
# fig = Figure(size=(800, 400))
# ax = Axis(fig[1, 1], xlabel="x", ylabel="h(x)")
# scatter!(ax, x[1:end-1], dx)
# fig

#####
##### Grid setup
#####
#
# Build the immersed-boundary grid for the rough Rayleigh–Bénard /
# sea-ice-formation simulation. Same topography, resolution, and domain size as
# `rough_rayleigh_benard_seaiceformation_noslip_halfdomain.jl`, but always
# periodic in x (topology = (Periodic, Flat, Bounded)).
#
#     grid = setup_grid(; Nr=16)

function setup_grid(; Nr, arch = GPU(), N = 512)

    Lx = 1
    Lz = 1
    Nx = N
    Nz = N

    # Periodic-in-x grid
    grid = RectilinearGrid(arch, Float64,
                           size = (Nx, Nz),
                           halo = (6, 6),
                           x = (0, Lx),
                           z = (0, Lz),
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

function setup_stretched_grid(; Nr, arch = GPU(), Nx = 512, Nz = 256)
    Lx = 1
    Lz = 1

    # Grid-spacing profile: fine spacing near the boundaries (where the thermal
    # boundary layers and roughness live) and coarser spacing in the interior.
    # Only the ratio d/c matters, since stretched_grid_from_spacing normalizes
    # the cumulative density.
    a = 50
    b = 50
    c = 1 / (2Nz)
    d = 1 / (Nz/2)
    f = 0.05
    g = 0.45

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
# Build the model on a previously constructed `grid` (see `setup_grid`). Same
# boundary conditions, closure, buoyancy, tracers, and initial conditions as the
# original script.
#
# `pressure_solver` is a builder `grid -> solver` so the solver can be
# constructed from the grid, e.g.
#
#     grid  = setup_grid(; Nr=16)
#     model = setup_model(grid, grid -> FFTBasedPoissonSolver(grid.underlying_grid); Ra=1e8)
#
#     model = setup_model(grid; Ra=1e8) do grid
#         reduced_precision_grid = with_number_type(Float32, grid.underlying_grid)
#         preconditioner = FFTBasedPoissonSolver(reduced_precision_grid)
#         ConjugateGradientPoissonSolver(grid, maxiter=80; preconditioner)
#     end
#
# An already-constructed solver object may also be passed directly.

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
