"""
    ceil_odd(x::T) where T<:Real

Ceil a number to the nearest odd integer.

# Arguments
- `x::T`: Number to ceil.

# Returns
- `Integer`: Nearest odd integer equal to or above `x`.
"""
ceil_odd(x::T) where T<:Real = Integer(ceil((x - 1) / 2) * 2 + 1)

"""
    insert_dim(x::T; pos::Integer) where T<:AbstractArray

# Arguments
- `x::T`: Array to insert dimension into.

# Keywords
- `pos::Integer`: Position of the new dimension.

# Returns
- `T`: `x` with an extra dimension at position `pos`.
"""
function insert_dim(x::T; pos::Integer) where T<: AbstractArray
    return reshape(x, size(x)[1:pos - 1]..., 1, size(x)[pos:end]...)
end

"""
    rbf(dist2::T) where T<:Real

# Arguments
- `dist2::T`: Squared distance.

# Returns
- `T`: RBF kernel evaluated at squared distance `dist2`.
"""
rbf(dist2::T) where T<:Real = exp(-0.5 * dist2)

"""
    compute_dists2(x::AbstractArray{T, 3}, y::AbstractArray{T, 3}) where T<:Real

Compute batched pairwise squared distances between 3-tensors `x` and `y`. The batch
dimension is the last dimension.

# Arguments
- `x::T`: Elements that correspond to the rows in the matrices of pairwise distances.
- `y::T`: Elements that correspond to the columns in the matrices of pairwise distances.

# Returns:
- `T`: Pairwise distances between and `x` and `y`.
"""
function compute_dists2(
    x::AbstractArray{T, 3},
    y::AbstractArray{T, 3}
) where T<:Real
    compute_dists2(x, y, Val(size(x, 2)))
end

function compute_dists2(
    x::AbstractArray{T, 3},
    y::AbstractArray{T, 3},
    ::Val{1}
) where T<:Real
    return (x .- permutedims(y, (2, 1, 3))).^2
end

function compute_dists2(
    x::AbstractArray{T, 3},
    y::AbstractArray{T, 3},
    d::Val
) where T<:Real
    y = permutedims(y, (2, 1, 3))
    return sum(x.^2; dims=2) .+ sum(y.^2; dims=1) .- 2 .* batched_mul(x, y)
end
