export SetConv, set_conv

"""
    SetConv{T<:AbstractVector{<:Real}}

A set convolution layer.

# Fields
- `log_scales::T`: Natural logarithm of the length scales of every input channel.
- `density:Bool`: Employ a density channel.
"""
struct SetConv{T<:AbstractVector{<:Real}}
    log_scales::T
    density::Bool
end

@Flux.treelike SetConv

"""
    set_conv(in_channels::Integer, scale::Float32; density::Bool=true)

Construct a set convolution layer.

# Arguments
- `in_channels::Integer`: Number of input channels.
- `scale::Real`: Initialisation of the length scales.

# Keywords
- `density:Bool`: Employ a density channel. This increases the number of output channels.

# Returns
- `SetConv`: Corresponding set convolution layer.
"""
function set_conv(in_channels::Integer, scale::Float32; density::Bool=true)
    # Add one to `in_channels` to account for the density channel.
    density && (in_channels += 1)
    scales = scale .* ones(Float32, in_channels)
    return SetConv(param(log.(scales)), density)
end

"""
    (layer::SetConv)(
        x_context::AbstractArray,
        y_context::AbstractArray,
        x_target::AbstractArray,
    )

# Arguments
- `x_context::AbstractArray`: Locations of observed values of shape `(n, d, batch)`.
- `y_context::AbstractArray`: Observed values of shape `(n, channels, batch)`.
- `x_target::AbstractArray`: Discretisation locations of shape `(m, d, batch)`.

# Returns
- `AbstractArray`: Output of layer of shape `(m, channels, batch)` or
    `(m, channels + 1, batch)`.
"""
function (layer::SetConv)(
    x_context::AbstractArray,
    y_context::AbstractArray,
    x_target::AbstractArray,
)
    n_context = size(x_context, 1)
    dimensionality = size(x_context, 2)
    batch_size = size(x_context, 3)

    # Validate input sizes.
    @assert size(y_context, 1) == n_context
    @assert size(x_target, 2) == dimensionality
    @assert size(y_context, 3) == batch_size
    @assert size(x_target, 3) == batch_size

    # Shape: `(n, m, batch)`.
    dists2 = compute_dists2(x_context, x_target)

    # Add channel dimension.
    # Shape: `(n, m, channels, batch)`.
    dists2 = insert_dim(dists2; pos=3)

    # Apply length scales.
    # Shape: `(n, m, channels, batch)`.
    scales = reshape(exp.(layer.log_scales), 1, 1, length(layer.log_scales), 1)
    dists2 = dists2 ./ scales.^2

    # Apply RBF to compute weights.
    weights = rbf(dists2)

    if layer.density
        # Add density channel to `y`.
        # Shape: `(n, channels + 1, batch)`.
        density = gpu(ones(eltype(y_context), n_context, 1, batch_size))
        channels = cat(density, y_context; dims=2)
    else
        channels = y_context
    end

    # Multiply with weights and sum.
    # Shape: `(m, channels + 1, batch)`.
    channels = dropdims(sum(insert_dim(channels; pos=2) .* weights; dims=1); dims=1)

    if layer.density
        # Divide by the density channel.
        density = channels[:, 1:1, :]
        others = channels[:, 2:end, :] ./ (density .+ 1f-8)
        channels = cat(density, others; dims=2)
    end

    return channels
end

function _to_rank_3(x)
    size_x = size(x)
    return reshape(x, size_x[1:2]..., prod(size_x[3:end])), function (y)
        return reshape(y, size(y)[1:2]..., size_x[3:end]...)
    end
end

_batched_mul(x, y) = Tracker.track(_batched_mul, x, y)

_transpose(x) = Tracker.track(_transpose, x)

__transpose(x) = permutedims(x, (2, 1, range(3, length(size(x)), step=1)...))

@Tracker.grad function _transpose(x)
    x = Tracker.data(x)
    return __transpose(x), ȳ -> (__transpose(ȳ),)
end

@Tracker.grad function _batched_mul(x, y)
    x, y = Tracker.data.((x, y))
    x, back = _to_rank_3(x)
    y, _ = _to_rank_3(y)
    return back(batched_mul(x, y)), function (ȳ)
        ȳ, _ = _to_rank_3(ȳ)
        return back(batched_mul(ȳ, __transpose(y))), back(batched_mul(__transpose(x), ȳ))
    end
end

function kernel(
    layer::SetConv,
    x_context::AbstractArray,
    y_context::AbstractArray,
    x_target::AbstractArray,
)
    n_context = size(x_context, 1)
    n_target = size(x_target, 1)
    dimensionality = size(x_context, 2)
    batch_size = size(x_context, 3)

    # Validate input sizes.
    @assert size(y_context, 1) == n_context
    @assert size(x_target, 2) == dimensionality
    @assert size(y_context, 3) == batch_size
    @assert size(x_target, 3) == batch_size

    # Shape: `(n, m, batch)`.
    dists2 = compute_dists2(x_context, x_target)

    # Add channel dimension.
    # Shape: `(n, m, channels, batch)`.
    dists2 = insert_dim(dists2; pos=3)

    # Apply length scales.
    # Shape: `(n, m, channels, batch)`.
    scales = reshape(exp.(layer.log_scales), 1, 1, length(layer.log_scales), 1)
    dists2 = dists2 ./ scales.^2

    # Apply RBF to compute weights.
    # Shape: `(n, m, channels, batch)`.
    weights = rbf(dists2)

    if layer.density
        # Add density channel to `y`.
        # Shape: `(n, channels + 1, batch)`.
        density = gpu(ones(eltype(y_context), n_context, 1, batch_size))
        channels = cat(density, y_context; dims=2)
    else
        channels = y_context
    end

    # Add target dimenion.
    # Shape: `(n, 1, channels + 1, batch)`.
    channels = insert_dim(channels; pos=2)

    # Multiply with weights and sum.
    # Shape: `(m, m, channels + 1, batch)`.
    channels = _batched_mul(_transpose(channels .* weights), weights)

    if layer.density
        # Divide by the density channel.
        density = channels[:, :, 1:1, :]
        others = channels[:, :, 2:end, :] ./ (density .+ 1f-8)
        channels = cat(density, others; dims=3)
    end

    # Add identity channel.
    identity = gpu(repeat(Matrix{Float32}(I, n_target, n_target), 1, 1, 1, batch_size))
    channels = cat(channels, identity; dims=3)

    return channels
end

function kernel_smooth(
    layer::SetConv,
    x_context::AbstractArray,
    y_context::AbstractArray,
    x_target::AbstractArray,
)
    n_context = size(x_context, 1)
    dimensionality = size(x_context, 2)
    batch_size = size(x_context, 3)

    # Validate input sizes.
    @assert size(y_context, 1) == n_context
    @assert size(y_context, 2) == n_context
    @assert size(x_target, 2) == dimensionality
    @assert size(y_context, 4) == batch_size
    @assert size(x_target, 3) == batch_size

    # Shape: `(n, m, batch)`.
    dists2 = compute_dists2(x_context, x_target)

    # Add channel dimension.
    # Shape: `(n, m, channels, batch)`.
    dists2 = insert_dim(dists2; pos=3)

    # Apply length scales.
    # Shape: `(n, m, channels, batch)`.
    scales = reshape(exp.(layer.log_scales), 1, 1, length(layer.log_scales), 1)
    dists2 = dists2 ./ scales.^2

    # Apply RBF to compute weights.
    # Shape: `(n, m, channels, batch)`.
    weights = rbf(dists2)

    # Multiply with weights and sum.
    # Shape: `(m, m, channels, batch)`.
    L = _batched_mul(y_context, weights)
    channels = _batched_mul(_transpose(L), L)

    return channels
end
