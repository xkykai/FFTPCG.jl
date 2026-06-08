using Interpolations: linear_interpolation, Line

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

"""
    stretched_grid_from_spacing(h, a, b, N; M=10_000)

Return `N` node positions on `[a, b]` whose local spacing follows the profile
`h`, i.e. nodes are clustered where `h` is small and spread out where `h` is
large.

The grid is built by mapping uniform points in a computational coordinate
`ξ ∈ [0, 1]` to physical space through the inverse of the cumulative grid
density `ρ = 1 / h`. `M` sets the resolution of the fine auxiliary grid used to
compute the cumulative integral.
"""
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
