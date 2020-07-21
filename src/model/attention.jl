export attention

"""
    Attention

# Fields
- `encoder_x`: Encoder that transforms inputs into keys or queries.
- `encoder_xy`: Encoder that transforms inputs and outputs into values.
- `mixer`: Linear transform that mixes the heads together.
- `transformer`: Completion of the transformer architecture.
"""
struct Attention
    encoder_x
    encoder_xy
    mixer
    transformer
end

@Flux.functor Attention

function code(layer::Attention, xz::AA, z::AA, x::AA; kws...)
    # Perform encodings.
    keys = layer.encoder_x(xz)
    queries = layer.encoder_x(x)
    values = layer.encoder_xy(cat(xz, z, dims=2))

    # Perform attention mechanism.
    products = batched_mul(queries, batched_transpose(keys))
    products = products ./ Float32(sqrt(size(queries, 2)))  # Keep variance constant.
    channels = batched_mul(softmax(products, dims=2), values)

    # Mix heads.
    channels = layer.mixer(channels)

    # Finish transformer architecture.
    return x, layer.transformer(channels, queries)
end

function code(layer::Attention, xz::Nothing, z::Nothing, x::AA; kws...)
    batch_size = size(x, 3)
    return x, zeros(Float32, 1, layer.transformer.ff₂.dim_out, batch_size) |> gpu
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
    struct Transformer

Completion of the transformer architecture.

# Fields
- `ff₁`: Feed-forward net to apply to the queries.
- `ln₁`: First layer normalisation layer.
- `ff₂`: Feed-forward net in the residual block.
- `ln₂`: Second layer normalisation layer.
"""
struct Transformer
    ff₁
    ln₁
    ff₂
    ln₂
end

@Flux.functor Transformer

"""
    transformer(dim_embedding::Integer, dim_head::Integer, num_heads::Integer)

# Arguments
- `dim_embedding::Integer`: Dimensionality of the embedding.
- `dim_heads::Integer`: Dimensionality of a head.
- `num_heads::Integer`: Number of heads.

# Returns
- `Transformer`: Corresponding layer.
"""
function transformer(dim_embedding::Integer, dim_head::Integer, num_heads::Integer)
    return Transformer(
        Chain(
            _compress_channels,
            batched_mlp(
                dim_in    =dim_head * num_heads,
                dim_hidden=dim_embedding,
                dim_out   =dim_embedding,
                num_layers=1
            )
        ),
        layer_norm(1, dim_embedding, 1),
        batched_mlp(
            dim_in    =dim_embedding,
            dim_hidden=dim_embedding,
            dim_out   =dim_embedding,
            num_layers=2
        ),
        layer_norm(1, dim_embedding, 1)
    )
end

"""
    (layer::Transformer)(channels::AA, queries::AA)

# Arguments
- `channels::AA`: Mixed heads.
- `queries::AA`: Queries. One per head.

# Returns
- `AA`: Output of transformer architecture.
"""
function (layer::Transformer)(channels::AA, queries::AA)
    channels = layer.ln₁(channels .+ layer.ff₁(queries))
    channels = layer.ln₂(channels .+ layer.ff₂(channels))
    return channels
end

"""
    attention(;
        dim_x::Integer,
        dim_y::Integer,
        dim_embedding::Integer,
        num_heads::Integer,
        num_encoder_layers::Integer=3
    )

Create an attention layer.

# Keywords
- `dim_x::Integer`: Dimensionality of the inputs.
- `dim_y::Integer`: Dimensionality of the outputs.
- `dim_embedding::Integer`: Dimensionality of the encodings.
- `num_heads::Integer`: Number of heads.
- `num_encoder_layers::Integer=3`: Number of layers for the value encoder.
"""
function attention(;
    dim_x::Integer,
    dim_y::Integer,
    dim_embedding::Integer,
    num_heads::Integer,
    num_encoder_layers::Integer=3
)
    dim_head = div(dim_embedding, num_heads)
    return Attention(
        Chain(
            batched_mlp(
                dim_in    =dim_x,
                dim_hidden=dim_head * num_heads,
                dim_out   =dim_head * num_heads,
                num_layers=num_encoder_layers
            ),
            x -> _extract_channels(x, num_heads)
        ),
        Chain(
            batched_mlp(
                dim_in    =dim_x + dim_y,
                dim_hidden=dim_head * num_heads,
                dim_out   =dim_head * num_heads,
                num_layers=num_encoder_layers
            ),
            x -> _extract_channels(x, num_heads)
        ),
        Chain(
            _compress_channels,
            batched_mlp(
                dim_in    =dim_head * num_heads,
                dim_hidden=dim_embedding,
                dim_out   =dim_embedding,
                num_layers=1
            )
        ),
        transformer(dim_embedding, dim_head, num_heads)
    )
end
