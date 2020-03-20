export Discretisation, UniformDiscretisation1d

"""
    Discretisation

Abstract type of all discretisation types.
"""
abstract type Discretisation end

"""
    UniformDiscretisation1d <: Discretisation

Uniform discretisation for one-dimensional inputs. It computes the minimum and maximum
input and determines a discretisation uniformly spanning this range at a specified
density.

This type does not need to be performant, which is why it is abstractly typed.

# Fields
- `poins_per_unit::Real`: Point per unit.
- `margin::Real`: Keep this amount as a margin of both sides of the maximum and minimum
    input.
- `multiple::Integer`: The number of discretisation points must be a multiple of this.
"""
struct UniformDiscretisation1d <: Discretisation
    points_per_unit::Real
    margin::Real
    multiple::Integer
end

"""
    (d::UniformDiscretisation1d)(xs::AbstractArray{T, 3}...) where T<:Real

# Arguments
- `xs::AbstractArray{T, 3}...`: Inputs to compute the discretisation for.

# Returns
- `T`: Discretisation.
"""
function (d::UniformDiscretisation1d)(xs::AbstractArray...)
    x = cat(xs...; dims=1)
    range_lower = minimum(x) - d.margin
    range_upper = maximum(x) + d.margin
    num_points = (range_upper - range_lower) * d.points_per_unit + 1
    num_points = ceil(num_points / d.multiple) * d.multiple
    disc = range(range_lower, range_upper, length=Integer(num_points))
    disc = gpu(collect(disc))
    return repeat(disc, 1, 1, size(x, 3))
end
