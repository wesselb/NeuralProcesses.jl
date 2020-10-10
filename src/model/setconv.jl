export set_conv

"""
    SetConv{T<:AV{<:Real}}

A set convolution layer.

# Fields
- `log_scales::T`: Natural logarithm of the length scales of every input channel.
- `density::Bool`: Include the density channel.
"""
struct SetConv{T<:AV{<:Real}}
    log_scales::T
    density::Bool
end

@Flux.functor SetConv

"""
    SetConvPD{T<:AV{<:Real}}

A set convolution layer for positive-definite-matrix outputs.

# Fields
- `log_scales::T`: Natural logarithm of the length scales of every input channel.
- `density::Bool`: Include the density channel.
"""
struct SetConvPD{T<:AV{<:Real}}
    log_scales::T
    density::Bool
end

@Flux.functor SetConvPD

"""
    set_conv(num_channels::Integer, scale::Float32; density::Bool=false)

Construct a set convolution layer.

# Arguments
- `num_channels::Integer`: Number of inputs channels, excluding the density channel.
- `scale::Float32`: Initialisation of the length scales.

# Keywords
- `density::Bool=false`: Include the density channel.
- `pd::Bool=false`: Generate positive-definite matrices as outputs.

# Returns
- `SetConv`: Corresponding set convolution layer.
"""
function set_conv(
    num_channels::Integer,
    scale::Float32;
    density::Bool=false,
    pd::Bool=false
)
    density && (num_channels += 1)
    scales = scale .* ones(Float32, num_channels)
    return (pd ? SetConvPD : SetConv)(log.(scales), density)
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
    density = ones_gpu(data_eltype(z), n, 1, batch_size)
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

function code(layer::SetConv, xz::AA, z::AA, x::AA; kws...)
    weights = _compute_weights(x, xz, _get_scales(layer))
    layer.density && (z = _prepend_density_channel(z))
    if ndims(z) == 4
        z = permutedims(z, (1, 4, 2, 3))
        z = batched_mul(weights, z)
        z = permutedims(z, (1, 3, 4, 2))
    elseif ndims(z) == 3
        z = with_dummy(c -> batched_mul(weights, c), z)
    else
        error("Cannot deal with inputs of rank $(ndims(z)).")
    end
    layer.density && (z = _normalise_by_first_channel(z))
    return x, z
end

function code(layer::SetConv, xz::Nothing, z::Nothing, x::AA; kws...)
    return x, zeros_gpu(
        data_eltype(x),
        size(x, 1),               # Size of encoding
        length(layer.log_scales), # Number of z, including the density channel
        size(x, 3)                # Batch size
    )
end

function code(layer::SetConvPD, xz::AA, z::AA, x::AA; kws...)
    weights = _compute_weights(x, xz, _get_scales(layer))
    # TODO: Disengtangle the below.
    if layer.density
        z = insert_dim(_prepend_density_channel(z), pos=1)
        z = batched_mul(weights .* z, batched_transpose(weights))
        z = _normalise_by_first_channel(z)
        z = _prepend_identity_channel(z)
    else
        Ls = batched_mul(weights, z)
        z = batched_mul(Ls, batched_transpose(Ls))
    end
    return x, z
end

function code(layer::SetConvPD, xz::Nothing, z::Nothing, x::AA; kws...)
    z = zeros_gpu(
        data_eltype(x),
        size(x, 1),               # Size of encoding
        size(x, 1),               # Again size of encoding: encoding is square
        length(layer.log_scales), # Number of z, including the density channel
        size(x, 3)                # Batch size
    )
    # Also prepend density channel.
    layer.density && (z = _prepend_identity_channel(z))
    return x, z
end
