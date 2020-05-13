export Attention, attention

"""
    Attention

# Fields
- `encoder_x`: Encoder that transforms inputs into keys.
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

@Flux.treelike Attention

"""
    (layer::Attention)(xc, yc, xt)

# Arguments
- `xc`: Locations of context set of shape `(n, dims, batch)`.
- `yc`: Observed values of context set of shape `(n, channels, batch)`.
- `xt`: Locations of target set of shape `(m, dims, batch)`.

# Returns
- `AbstractArray`: Encodings of shape `(m, dim_embedding, batch)`.
"""
function (layer::Attention)(xc, yc, xt)
    # Perform encodings.
    keys = layer.encoder_x(xc)
    queries = layer.encoder_x(xt)
    values = layer.encoder_xy(cat(xc, yc, dims=2))

    # Perform attention mechanism.
    weights = exp.(batched_mul(queries, batched_transpose(keys)))
    channels = batched_mul(weights, values) ./ sum(weights, dims=2)

    # Mix heads.
    channels = layer.mixer(channels)

    # Finish transformer architecture.
    return layer.transformer(channels, queries)
end

"""
    empty_encoding(layer::Attention, xt)

Construct an encoding for the empty set.

# Arguments
- `layer::Attention`: Layer.
- `xt`: Locations of target set of shape `(m, dims, batch)`.

# Returns
- `AbstractArray`: Empty encoding.
"""
function empty_encoding(layer::Attention, xt)
    m, _, batch_size = size(xt)
    return zeros(Float32, m, layer.transformer.ff₂.dim_out, batch_size) |> gpu
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

@Flux.treelike Transformer

"""
    Transformer(dim_embedding::Integer, num_heads::Integer)

# Arguments
- `dim_embedding::Integer`: Dimensionality of the embedding.
- `num_heads::Integer`: Number of heads.

# Returns
- `Transformer`: Corresponding layer.
"""
function Transformer(dim_embedding::Integer, num_heads::Integer)
    return Transformer(
        Chain(
            _compress_channels,
            batched_mlp(
                dim_in    =dim_embedding * num_heads,
                dim_hidden=dim_embedding,
                dim_out   =dim_embedding,
                num_layers=1
            )
        ),
        LayerNorm(1, dim_embedding, 1),
        batched_mlp(
            dim_in    =dim_embedding,
            dim_hidden=dim_embedding,
            dim_out   =dim_embedding,
            num_layers=2
        ),
        LayerNorm(1, dim_embedding, 1)
    )
end

"""
    (layer::Transformer)(channels, queries)

# Arguments
- `channels`: Mixed heads.
- `queries`: Queries. One per head.

# Returns
- `AbstractArray`: Output of transformer architecture.
"""
function (layer::Transformer)(channels, queries)
    channels = layer.ln₁(channels .+ layer.ff₁(queries))
    channels = layer.ln₂(channels .+ layer.ff₂(channels))
    return channels
end

"""
    struct LayerNorm

Layer normalisation.

# Fields
- `w`: Weights.
- `b`: Biases.
- `dims`: Dimensions to apply the normalisation to.
"""
struct LayerNorm
    w
    b
    dims
end

@Flux.treelike LayerNorm

"""
    LayerNorm(shape...)

Construct a `LayerNorm` layer.

# Arguments
- `shape...`: A tuple containing one integer per dimension. Set a dimension to `1` to not
    normalise or set a dimenion to the size of that dimension to do normalise.

# Returns
- `LayerNorm`: Corresponding layer.
"""
function LayerNorm(shape...)
    return LayerNorm(
        param(ones(Float32, shape...)),
        param(zeros(Float32, shape...)),
        findall(x -> x > 1, shape)
    )
end

"""
    (layer::LayerNorm)(x)

# Arguments
- `x`: Unnormalised input.

# Returns
- `AbstractArray`: `x` normalised.
"""
function (layer::LayerNorm)(x)
    x = x .- mean(x, dims=layer.dims)
    x = x ./ sqrt.(mean(x .* x, dims=layer.dims) .+ 1f-8)
    return x .* layer.w .+ layer.b
end

"""
    struct BatchedMLP

# Fields
- `mlp`: MLP to batch.
- `dim_out::Integer`: Dimensionality of the output.
"""
struct BatchedMLP
    mlp
    dim_out::Integer
end

@Flux.treelike BatchedMLP

"""
    (layer::BatchedMLP)(x)

# Arguments
- `x`: Batched input.

# Returns
- `AbstractArray`: Result of applying `layer.mlp` to every batch in `x`.
"""
function (layer::BatchedMLP)(x)
    x, back = _to_rank_3(x)
    x = with_dummy(layer.mlp, x)
    return back(x)
end

"""
    batched_mlp(;
        dim_in::Integer,
        dim_hidden::Integer=dim_in,
        dim_out::Integer,
        num_layers::Integer
    )

Construct a batched MLP.

# Keywords
- `dim_in::Integer`: Dimensionality of the input.
- `dim_hidden::Integer=dim_in`: Dimensionality of the hidden layers.
- `dim_out::Integer`: Dimensionality of the output.
- `num_layers::Integer`: Number of layers.

# Returns
- `BatchedMLP`: Corresponding batched MLP.
"""
function batched_mlp(;
    dim_in::Integer,
    dim_hidden::Integer=dim_in,
    dim_out::Integer,
    num_layers::Integer,
)
    act(x) = leakyrelu(x, 0.01f0)  # Use a small leak here.
    if num_layers == 1
        return BatchedMLP(_dense(dim_in, dim_out), dim_out)
    else
        layers = Any[_dense(dim_in, dim_hidden, act)]
        for i = 1:num_layers - 2
            push!(layers, _dense(dim_hidden, dim_hidden, act))
        end
        push!(layers, _dense(dim_hidden, dim_out))
        return BatchedMLP(Chain(layers...), dim_out)
    end
end

_dense(dim_in, dim_out, args...) =
    Conv(Flux.param.(_init_conv_random_bias((1, 1), dim_in=>dim_out))..., args...)

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
    return Attention(
        Chain(
            batched_mlp(
                dim_in    =dim_x,
                dim_hidden=dim_embedding,
                dim_out   =dim_embedding * num_heads,
                num_layers=1
            ),
            x -> _extract_channels(x, num_heads)
        ),
        Chain(
            batched_mlp(
                dim_in    =dim_x + dim_y,
                dim_hidden=dim_embedding,
                dim_out   =dim_embedding * num_heads,
                num_layers=num_encoder_layers
            ),
            x -> _extract_channels(x, num_heads)
        ),
        Chain(
            _compress_channels,
            batched_mlp(
                dim_in    =dim_embedding * num_heads,
                dim_hidden=dim_embedding,
                dim_out   =dim_embedding,
                num_layers=1
            )
        ),
        Transformer(dim_embedding, num_heads)
    )
end
