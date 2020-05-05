export Attention, attention, BatchedLinear

"""
    Attention

# Fields
- `encoder_x`: Encoder that transforms inputs into keys.
- `encoder_xy`: Encoder that transforms inputs and outputs into values.
- `mixer`: Linear transform that mixes the heads together.
"""
struct Attention
    encoder_x
    encoder_xy
    mixer
end

@Flux.treelike Attention

"""
    (layer::Attention)(
        x_context::AbstractArray,
        y_context::AbstractArray,
        x_target::AbstractArray
    )

# Arguments
- `x_context::AbstractArray`: Locations of observed values of shape `(n, dim_x, batch)`.
- `y_context::AbstractArray`: Observed values of shape `(n, dim_y, batch)`.
- `x_target::AbstractArray`: Locations of target values of shape `(m, dim_x, batch)`.

# Returns
- `AbstractArray`: Encodings of shape `(m, dim_embedding, batch)`.
"""
function (layer::Attention)(
    x_context::AbstractArray,
    y_context::AbstractArray,
    x_target::AbstractArray
)
    # Perform encodings.
    keys = layer.encoder_x(x_context)
    queries = layer.encoder_x(x_target)
    values = layer.encoder_xy(cat(x_context, y_context, dims=2))

    # Perform attention mechanism.
    weights = exp.(batched_mul(queries, batched_transpose(keys)))
    channels = batched_mul(weights, values) ./ sum(weights, dims=2)

    # Mix all heads.
    return layer.mixer(channels)
end

function _extract_channels(x, num_channels)
    n, _, batch_size = size(x)
    return reshape(x, n, :, num_channels, batch_size)
end

function _compress_channels(x)
    n, _, _, batch_size = size(x)
    return reshape(x, n, :, batch_size)
end

"""
    attention(;
        dim_x::Integer,
        dim_y::Integer,
        dim_embedding::Integer,
        num_channels::Integer
    )

Create an attention layer.

# Keywords
- `dim_x::Integer`: Dimensionality of the inputs.
- `dim_y::Integer`: Dimensionality of the outputs.
- `dim_embedding::Integer`: Dimensionality of the encodings.
- `num_channels::Integer`: Number of heads.
"""
function attention(;
    dim_x::Integer,
    dim_y::Integer,
    dim_embedding::Integer,
    num_channels::Integer
)
    act(x) = leakyrelu.(x, 0.1f0)
    return Attention(
        Chain(
            BatchedLinear(dim_x, dim_embedding * num_channels),
            act,
            # First extract channels to make the subsequent linear layer channel-wise.
            x -> _extract_channels(x, num_channels),
            BatchedLinear(dim_embedding),
        ),
        Chain(
            BatchedLinear(dim_x + dim_y, dim_embedding * num_channels),
            act,
            x -> _extract_channels(x, num_channels),  # See above.
            BatchedLinear(dim_embedding),
        ),
        Chain(
            _compress_channels,
            BatchedLinear(dim_embedding * num_channels, dim_embedding)
        )
    )
end

"""
    BatchedLinear

# Fields
- `w`: Weights.
- `b`: Biases.
"""
struct BatchedLinear
    w
    b
end

"""
    BatchedLinear(dim_in::Integer, dim_out::Integer=dim_in)

# Arguments
- `dim_in::Integer`: Input dimensionality.
- `dim_out::Integer=dim_in`: Output dimensionality.

# Returns
- `BatchedLinear`: Corresponding layer.
"""
function BatchedLinear(dim_in::Integer, dim_out::Integer=dim_in)
    return BatchedLinear(
        Flux.param(Flux.glorot_normal(dim_in, dim_out)),
        Flux.param(Flux.glorot_normal(1, dim_out))
    )
end

@Flux.treelike BatchedLinear

"""
    (layer::BatchedLinear)(x)

# Arguments
- `x::AbstractArray`: Batched inputs.

# Returns
- `AbstractArray`: Batched output of linear layer.
"""
function (layer::BatchedLinear)(x::AbstractArray)
    x, back = _to_rank_3(x)  # Merge all batch dimensions.

    n, _, batch_size = size(x)

    # Merge batch dimension into data dimension.
    x = reshape(permutedims(x, (1, 3, 2)), n * batch_size, :)

    # Perform linear layer.
    x = batched_mul(x, layer.w) .+ layer.b

    # Separate batch dimension from data dimension.
    x = permutedims(reshape(x, n, batch_size, :), (1, 3, 2))

    return back(x)  # Unmerge batch dimensions.
end
