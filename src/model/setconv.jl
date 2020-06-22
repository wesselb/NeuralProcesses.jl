export set_conv

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
- `num_channels::Integer`: Number of input z. This should include the density
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

function _prepend_density_channel(z)
    n, _, batch_size = size(z)
    density = ones_gpu(eltype(z), n, 1, batch_size)
    return cat(density, z, dims=2)
end

function _prepend_identity_channel(z)
    n, _, _, batch_size = size(z)
    identity = gpu(repeat(Matrix{Float32}(I, n, n), 1, 1, 1, batch_size))
    return cat(z, identity, dims=3)
end

function _normalise_by_first_channel(z)
    channels_dim = ndims(z) - 1  # Channels dimension is second to last.
    normaliser = slice_at(z, channels_dim, 1:1)
    others = slice_at(z, channels_dim, 2:size(z, channels_dim))
    return cat(normaliser, others ./ (normaliser .+ 1f-8), dims=channels_dim)
end

function encode(layer::SetConv, xz::AA, z::AA, x::AA; kws...)
    weights = _compute_weights(x, xz, _get_scales(layer))
    z = _prepend_density_channel(z)
    z = with_dummy(c -> batched_mul(weights, c), z)
    return x, _normalise_by_first_channel(z)
end

function encode(layer::SetConv, xz::Nothing, z::Nothing, x::AA; kws...)
    return x, zeros_gpu(
        eltype(x),
        size(x, 1),               # Size of encoding
        length(layer.log_scales), # Number of z, including the density channel
        size(x, 3)                # Batch size
    )
end

function decode(layer::SetConv, xz::AA, z::AA, x::AA)
    weights = _compute_weights(x, xz, _get_scales(layer))
    return x, with_dummy(c -> batched_mul(weights, c), z)
end

function encode_pd(layer::SetConv, xz::AA, z::AA, x::AA; kws...)
    weights = _compute_weights(x, xz, _get_scales(layer))
    z = insert_dim(_prepend_density_channel(z), pos=1)
    z = batched_mul(weights .* z, batched_transpose(weights))
    z = _normalise_by_first_channel(z)
    return xz, _prepend_identity_channel(z)
end

function encode_pd(layer::SetConv, xz::Nothing, z::Nothing, x::AA; kws...)
    return x, _prepend_identity_channel(zeros_gpu(  # Also prepend identity channel.
        eltype(x),
        size(x, 1),               # Size of encoding
        size(x, 1),               # Again size of encoding: encoding is square
        length(layer.log_scales), # Number of z, including the density channel
        size(x, 3)                # Batch size
    ))
end

function decode_pd(layer::SetConv, xz::AA, z::AA, x::AA)
    weights = _compute_weights(x, xz, _get_scales(layer))
    Ls = batched_mul(weights, z)
    return x, batched_mul(Ls, batched_transpose(Ls))
end
