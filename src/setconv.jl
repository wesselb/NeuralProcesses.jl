export SetConv, set_conv

"""
    SetConv{T<:AV{<:Real}}

A set convolution layer.

# Fields
- `log_scales::T`: Natural logarithm of the length scales of every input channel.
"""
struct SetConv{T<:AV{<:Real}}
    log_scales::T
end

@Flux.treelike SetConv

"""
    set_conv(num_channels::Integer, scale::Float32)

Construct a set convolution layer.

# Arguments
- `num_channels::Integer`: Number of input channels. This should include the density
    channel, which is automatically prepended upon encoding.
- `scale::Float32`: Initialisation of the length scales.

# Returns
- `SetConv`: Corresponding set convolution layer.
"""
function set_conv(num_channels::Integer, scale::Float32)
    scales = scale .* ones(Float32, num_channels)
    return SetConv(param(log.(scales)))
end

_get_scales(layer) = reshape(exp.(layer.log_scales), 1, 1, length(layer.log_scales), 1)

function _compute_weights(x, y, scales)
    dists² = compute_dists²(x, y)
    dists² = insert_dim(dists², pos=3)  # Add channel dimension.
    dists² = dists² ./ scales.^2
    return rbf(dists²)
end

function _prepend_density_channel(channels)
    n, _, batch_size = size(channels)
    density = ones_gpu(eltype(channels), n, 1, batch_size)
    return cat(density, channels, dims=2)
end

function _prepend_identity_channel(channels)
    n, _, _, batch_size = size(channels)
    identity = gpu(repeat(Matrix{Float32}(I, n, n), 1, 1, 1, batch_size))
    return cat(channels, identity, dims=3)
end

function _normalise_by_first_channel(channels)
    channel_dim = ndims(channels) - 1  # Channel dimension is second to last.
    normaliser = slice_at(channels, channel_dim, 1:1)
    others = slice_at(channels, channel_dim, 2:size(channels, channel_dim))
    return cat(normaliser, others ./ (normaliser .+ 1f-8), dims=channel_dim)
end

"""
    encode(layer::SetConv, xc::AA, yc::AA, xz::AA)

# Arguments
- `layer::SetConv`: Set convolution layer.
- `xc::AA`: Locations of context set of shape `(n, dims, batch)`.
- `yc::AA`: Observed values of context set of shape `(n, channels, batch)`.
- `xz::AA`: Locations of latent encoding of shape `(k, dims, batch)`.

# Returns
- `AA`: Output of layer of shape `(k, channels + 1, batch)`.
"""
function encode(layer::SetConv, xc::AA, yc::AA, xz::AA)
    weights = _compute_weights(xz, xc, _get_scales(layer))
    channels = _prepend_density_channel(yc)
    channels = with_dummy(c -> batched_mul(weights, c), channels)
    return _normalise_by_first_channel(channels)
end

"""
    empty_encoding(layer::SetConv, xz::AA)

# Arguments
- `layer::SetConv`: Set convolution layer.
- `xz`: Locations of latent encoding of shape `(k, dims, batch)`.

# Returns
- `AA`: All zeros output of shape `(k, channels + 1, batch)`.
"""
function empty_encoding(layer::SetConv, xz::AA)
    return zeros_gpu(
        eltype(xz),
        size(xz, 1),              # Size of encoding
        length(layer.log_scales), # Number of channels, including the density channel
        size(xz, 3)               # Batch size
    )
end

"""
    decode(layer::SetConv, xz::AA, channels::AA, xt::AA)

# Arguments
- `layer::SetConv`: Set convolution layer.
- `xz::AA`: Locations of latent encoding of shape `(k, dims, batch)`.
- `channels::AA`: Channels of shape `(k, channels, batch)`.
- `xt::AA`: Locations of target set of shape `(m, dims, batch)`.

# Returns
- `AA`: Output of layer of shape `(m, channels, batch)`.
"""
function decode(layer::SetConv, xz::AA, channels::AA, xt::AA)
    weights = _compute_weights(xt, xz, _get_scales(layer))
    return with_dummy(c -> batched_mul(weights, c), channels)
end

"""
    encode_pd(layer::SetConv, xc::AA, yc::AA, xz::AA)

# Arguments
- `layer::SetConv`: Set convolution layer.
- `xc::AA`: Locations of context set of shape `(n, dims, batch)`.
- `yc::AA`: Observed values of context set of shape `(n, channels, batch)`.
- `xz::AA`: Locations of latent encoding of shape `(k, dims, batch)`.

# Returns
- `AA`: Output of layer of shape `(k, k, channels + 2, batch)`.
"""
function encode_pd(layer::SetConv, xc::AA, yc::AA, xz::AA)
    weights = _compute_weights(xz, xc, _get_scales(layer))
    channels = insert_dim(_prepend_density_channel(yc), pos=1)
    channels = batched_mul(weights .* channels, batched_transpose(weights))
    channels = _normalise_by_first_channel(channels)
    return _prepend_identity_channel(channels)
end

"""
    empty_encoding_pd(layer::SetConv, xz::AA)

# Arguments
- `layer::SetConv`: Set convolution layer.
- `xz::AA`: Locations of latent encoding of shape `(k, dims, batch)`.

# Returns
- `AA`: Output of shape `(k, k, channels + 2, batch)`.
"""
function empty_encoding_pd(layer::SetConv, xz::AA)
    return _prepend_identity_channel(zeros_gpu(  # Also prepend identity channel.
        eltype(xz),
        size(xz, 1),              # Size of encoding
        size(xz, 1),              # Again size of encoding: encoding is square
        length(layer.log_scales), # Number of channels, including the density channel
        size(xz, 3)               # Batch size
    ))
end

"""
    decode_pd(layer::SetConv, xz::AA, channels::AA, xt::AA)

# Arguments
- `layer::SetConv`: Set convolution layer.
- `xz::AA`: Locations of latent encoding of shape `(k, dims, batch)`.
- `channels::AA`: Channels of shape `(k, k, channels, batch)`.
- `xt::AA`: Locations of target set of shape `(m, dims, batch)`.

# Returns
- `AA`: Output of layer of shape `(m, m, channels, batch)`.
"""
function decode_pd(layer::SetConv, xz::AA, channels::AA, xt::AA)
    weights = _compute_weights(xt, xz, _get_scales(layer))
    Ls = batched_mul(weights, channels)
    return batched_mul(Ls, batched_transpose(Ls))
end
