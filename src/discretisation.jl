export Discretisation, UniformDiscretisation1D

"""
    Discretisation

Abstract type of all discretisation types.
"""
abstract type Discretisation end

"""
    UniformDiscretisation1D <: Discretisation

Uniform discretisation for one-dimensional inputs. It computes the minimum and maximum
input and determines a discretisation uniformly spanning this range at a specified
density.

This type does not need to be performant, which is why it is abstractly typed.

# Fields
- `poins_per_unit::Float32`: Points per unit.
- `margin::Float32`: Keep this amount as a margin of both sides of the maximum and minimum
    input.
- `multiple::Integer=1`: The number of discretisation points must be a multiple of this.
"""
struct UniformDiscretisation1D <: Discretisation
    points_per_unit::Float32
    margin::Float32
    multiple::Integer
end

UniformDiscretisation1D(points_per_unit::Float32, margin::Float32) =
    UniformDiscretisation1D(points_per_unit, margin, 1)

"""
    (d::UniformDiscretisation1D)(xs::AA...)

# Arguments
- `xs...`: Inputs to compute the discretisation for.

# Returns
- `AA`: Discretisation.
"""
function (d::UniformDiscretisation1D)(xs...; margin=d.margin, kws...)
    x = cat(filter(x -> !isnothing(x), collect(xs))...; dims=1)
    range_lower = minimum(x) - margin
    range_upper = maximum(x) + margin
    num_points = (range_upper - range_lower) * d.points_per_unit + 1
    num_points = ceil(num_points / d.multiple) * d.multiple
    disc = collect(range(range_lower, range_upper, length=Integer(num_points))) |> gpu
    return repeat_gpu(disc, 1, 1, size(x, 3))  # Match batch size of input.
end
