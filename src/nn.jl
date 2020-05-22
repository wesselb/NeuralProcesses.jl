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
    dims::Tuple
end

@Flux.treelike LayerNorm

"""
    layer_norm(shape::Integer...)

Construct a `LayerNorm` layer.

# Arguments
- `shape...`: A tuple containing one integer per dimension. Set a dimension to `1` to not
    normalise. Set a dimension to the size of that dimension to do normalise.

# Returns
- `LayerNorm`: Corresponding layer.
"""
function layer_norm(shape::Integer...)
    return LayerNorm(
        param(ones(Float32, shape...)),
        param(zeros(Float32, shape...)),
        Tuple(findall(x -> x > 1, shape))
    )
end

"""
    (layer::LayerNorm)(x::AA)

# Arguments
- `x::AA`: Unnormalised input.

# Returns
- `AA`: `x` normalised.
"""
function (layer::LayerNorm)(x::AA)
    x = x .- mean(x, dims=layer.dims)
    x = x ./ sqrt.(mean(x .* x, dims=layer.dims) .+ 1f-8)
    return x .* layer.w .+ layer.b
end

"""
    struct BatchedMLP

# Fields
- `mlp`: MLP.
- `dim_out::Integer`: Dimensionality of the output.
"""
struct BatchedMLP
    mlp
    dim_out::Integer
end

@Flux.treelike BatchedMLP

"""
    (layer::BatchedMLP)(x::AA)

# Arguments
- `x::AA`: Batched input.

# Returns
- `AA`: Result of applying `layer.mlp` to every batch in `x`.
"""
function (layer::BatchedMLP)(x::AA)
    x, back = to_rank_3(x)  # Compress all batch dimensions.

    n, _, batch_size = size(x)

    # Merge data point and batch dimension.
    x = permutedims(x, (2, 1, 3))
    x = reshape(x, :, n * batch_size)

    x = layer.mlp(x)

    # Unmerge data point and batch dimension.
    x = reshape(x, :, n, batch_size)
    x = permutedims(x, (2, 1, 3))

    return back(x)  # Uncompress batch dimensions.
end

"""
    batched_mlp(;
        dim_in::Integer,
        dim_hidden::Integer=dim_in,
        dim_out::Integer,
        num_layers::Integer,
        act=x -> leakyrelu(x, 0.1f0)
    )

Construct a batched MLP.

# Keywords
- `dim_in::Integer`: Dimensionality of the input.
- `dim_hidden::Integer`: Dimensionality of the hidden layers.
- `dim_out::Integer`: Dimensionality of the output.
- `num_layers::Integer`: Number of layers.
- `act=x -> leakyrelu(x, 0.1f0)`: Activation function to use.

# Returns
- `BatchedMLP`: Corresponding batched MLP.
"""
function batched_mlp(;
    dim_in::Integer,
    dim_hidden::Integer,
    dim_out::Integer,
    num_layers::Integer,
    act=x -> leakyrelu(x, 0.1f0)
)
    if num_layers == 1
        return BatchedMLP(Chain(Dense(dim_in, dim_out)), dim_out)
    else
        layers = Any[Dense(dim_in, dim_hidden, act)]
        for i = 1:num_layers - 2
            push!(layers, Dense(dim_hidden, dim_hidden, act))
        end
        push!(layers, Dense(dim_hidden, dim_out))
        return BatchedMLP(Chain(layers...), dim_out)
    end
end
