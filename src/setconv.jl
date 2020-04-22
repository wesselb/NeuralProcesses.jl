export SetConv, set_conv,
    encode, empty_encoding, decode,
    encode_pd, empty_encoding_pd, decode_pd

"""
    SetConv{T<:AbstractVector{<:Real}}

A set convolution layer.

# Fields
- `log_scales::T`: Natural logarithm of the length scales of every input channel.
"""
struct SetConv{T<:AbstractVector{<:Real}}
    log_scales::T
end

@Flux.treelike SetConv

"""
    set_conv(num_channels::Integer, scale::Float32)

Construct a set convolution layer.

# Arguments
- `num_channels::Integer`: Number of input channels. This should include the density
    channel, which is automatically prepended upon encoding.
- `scale::Real`: Initialisation of the length scales.

# Returns
- `SetConv`: Corresponding set convolution layer.
"""
function set_conv(num_channels::Integer, scale::Float32)
    scales = scale .* ones(Float32, num_channels)
    return SetConv(param(log.(scales)))
end

_get_scales(layer) = reshape(exp.(layer.log_scales), 1, 1, length(layer.log_scales), 1)

function _compute_weights(x, y, scales)
    dists2 = compute_dists2(x, y)
    dists2 = insert_dim(dists2, pos=3)  # Add channel dimension.
    dists2 = dists2 ./ scales.^2
    return rbf(dists2)
end

function _prepend_density_channel(channels)
    n, _, batch_size = size(channels)
    density = gpu(ones(eltype(channels), n, 1, batch_size))
    return cat(density, channels, dims=2)
end

function _prepend_identity_channel(channels)
    n, _, _, batch_size = size(channels)
    identity = gpu(repeat(Matrix{Float32}(I, n, n), 1, 1, 1, batch_size))
    return cat(channels, identity, dims=3)
end

function _normalise_by_first_channel(channels)
    normaliser = channels[:, :, 1:1, :]
    others = channels[:, :, 2:end, :] ./ (normaliser .+ 1f-8)
    return cat(normaliser, others, dims=3)
end

"""
    encode(
        layer::SetConv,
        x_context::AbstractArray,
        y_context::AbstractArray,
        x_target::AbstractArray,
    )

# Arguments
- `layer::SetConv`: Set convolution layer.
- `x_context::AbstractArray`: Locations of observed values of shape `(n, d, batch)`.
- `y_context::AbstractArray`: Observed values of shape `(n, channels, batch)`.
- `x_target::AbstractArray`: Locations of target values of shape `(m, d, batch)`.

# Returns
- `AbstractArray`: Output of layer of shape `(m, 1, channels + 1, batch)`.
"""
function encode(
    layer::SetConv,
    x_context::AbstractArray,
    y_context::AbstractArray,
    x_target::AbstractArray;
)
    weights = _compute_weights(x_target, x_context, _get_scales(layer))
    channels = insert_dim(_prepend_density_channel(y_context), pos=2)
    channels = batched_mul(weights, channels)
    return _normalise_by_first_channel(channels)
end

"""
    empty_encoding(
        layer::SetConv,
        y_context::AbstractArray,
        x_target::AbstractArray,
    )

# Arguments
- `layer::SetConv`: Set convolution layer.
- `y_context::AbstractArray`: Observed values of shape `(n, channels, batch)`.
- `x_target::AbstractArray`: Locations of target values of shape `(m, d, batch)`.

# Returns
- `AbstractArray`: All zeros output of shape `(m, 1, channels + 1, batch)`.
"""
function empty_encoding(
    layer::SetConv,
    y_context::AbstractArray,
    x_target::AbstractArray
)
    return gpu(zeros(
        eltype(y_context),
        size(x_target, 1),        # Size of encoding
        1,                        # Required to make it a 2D convolution.
        length(layer.log_scales), # Number of channels, including the density channel
        size(y_context, 3)        # Batch size
    ))
end

"""
    decode(
        layer::SetConv,
        x_context::AbstractArray,
        y_context::AbstractArray,
        x_target::AbstractArray,
    )

# Arguments
- `layer::SetConv`: Set convolution layer.
- `x_context::AbstractArray`: Locations of observed values of shape `(n, d, batch)`.
- `y_context::AbstractArray`: Observed values of shape `(n, 1, channels, batch)`.
- `x_target::AbstractArray`: Locations of target values of shape `(m, d, batch)`.

# Returns
- `AbstractArray`: Output of layer of shape `(m, 1, channels, batch)`.
"""
function decode(
    layer::SetConv,
    x_context::AbstractArray,
    y_context::AbstractArray,
    x_target::AbstractArray;
)
    weights = _compute_weights(x_target, x_context, _get_scales(layer))
    return batched_mul(weights, y_context)
end

"""
    encode_pd(
        layer::SetConv,
        x_context::AbstractArray,
        y_context::AbstractArray,
        x_target::AbstractArray,
    )

# Arguments
- `layer::SetConv`: Set convolution layer.
- `x_context::AbstractArray`: Locations of observed values of shape `(n, d, batch)`.
- `y_context::AbstractArray`: Observed values of shape `(n, channels, batch)`.
- `x_target::AbstractArray`: Locations of target values of shape `(m, d, batch)`.

# Returns
- `AbstractArray`: Output of layer of shape `(m, m, channels + 2, batch)`.
"""
function encode_pd(
    layer::SetConv,
    x_context::AbstractArray,
    y_context::AbstractArray,
    x_target::AbstractArray,
)
    weights = _compute_weights(x_target, x_context, _get_scales(layer))
    channels = insert_dim(_prepend_density_channel(y_context), pos=1)
    channels = batched_mul(weights .* channels, batched_transpose(weights))
    channels = _normalise_by_first_channel(channels)
    return _prepend_identity_channel(channels)
end

"""
    empty_encoding_pd(
        layer::SetConv,
        y_context::AbstractArray,
        x_target::AbstractArray,
    )

# Arguments
- `layer::SetConv`: Set convolution layer.
- `y_context::AbstractArray`: Observed values of shape `(n, channels, batch)`.
- `x_target::AbstractArray`: Locations of target values of shape `(m, d, batch)`.

# Returns
- `AbstractArray`: All zeros output of shape `(m, m, channels + 2, batch)`.
"""
function empty_encoding_pd(
    layer::SetConv,
    y_context::AbstractArray,
    x_target::AbstractArray
)
    return _prepend_identity_channel(gpu(zeros(  # Also prepend identity channel
        eltype(y_context),
        size(x_target, 1),        # Size of encoding
        size(x_target, 1),        # Again size of encoding: encoding is square
        length(layer.log_scales), # Number of channels, including the density channel
        size(y_context, 3)        # Batch size
    )))
end

"""
    decode_pd(
        layer::SetConv,
        x_context::AbstractArray,
        y_context::AbstractArray,
        x_target::AbstractArray,
    )

# Arguments
- `layer::SetConv`: Set convolution layer.
- `x_context::AbstractArray`: Locations of observed values of shape `(n, d, batch)`.
- `y_context::AbstractArray`: Observed values of shape `(n, n, channels, batch)`.
- `x_target::AbstractArray`: Locations of target values of shape `(m, d, batch)`.

# Returns
- `AbstractArray`: Output of layer of shape `(m, m, channels, batch)`.
"""
function decode_pd(
    layer::SetConv,
    x_context::AbstractArray,
    y_context::AbstractArray,
    x_target::AbstractArray,
)
    weights = _compute_weights(x_target, x_context, _get_scales(layer))
    Ls = batched_mul(weights, y_context)
    return batched_mul(Ls, batched_transpose(Ls))
end
