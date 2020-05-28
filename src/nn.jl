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


"""
    struct SplitGlobal

# Fields
- `num_global_channels::Integer`: Number of channels to use for the global channels.
- `ff₁`: Feed-forward net before pooling.
- `pooling`: Pooling.
- `ff₂`: Feed-forward net after pooling.
- `transform`: Function that transforms both the local and global channel after `ff₂`.
"""
struct SplitGlobal
    num_global_channels::Integer
    ff₁
    pooling
    ff₂
    transform
end

@Flux.treelike SplitGlobal

"""
    (layer::SplitGlobal)(x::AA)

Split `layer.num_global_channels` off of `x` to construct global channels.

# Arguments
- `x::AA`: Tensor to split global channels off of.

# Returns
- `Tuple`: Two-tuple containing the outputs for the global and local channels.
"""
function (layer::SplitGlobal)(x::AA)
    # Split channels.
    x_global = slice_at(x, 2, 1:layer.num_global_channels)
    x_local = slice_at(x, 2, layer.num_global_channels + 1:size(x, 2))

    # Pool over data points to make global channels.
    x_global = layer.ff₁(x_global)
    x_global = layer.pooling(x_global)
    x_global = layer.ff₂(x_global)

    return layer.transform(x_global), layer.transform(x_local)
end

"""
    struct MeanPooling

Mean pooling.

# Fields
- `ln`: Layer normalisation to depend not on the size of the discretisation.
"""
struct MeanPooling
    ln
end

@Flux.treelike MeanPooling

"""
    (layer::MeanPooling)(x::AA)

# Arguments
- `x::AA`: Input to pool.

# Returns
- `AA`: `x` pooled.
"""
(layer::MeanPooling)(x::AA) = layer.ln(mean(x, dims=1))

"""
    struct SumPooling

Sum pooling.

# Fields
- `factor::Integer`: Factor to divide by after pooling to help initialisation.
"""
struct SumPooling
    factor::Integer
end

@Flux.treelike SumPooling

"""
    (layer::SumPooling)(x::AA)

# Arguments
- `x::AA`: Input to pool.

# Returns
- `AA`: `x` pooled.
"""
(layer::SumPooling)(x::AA) = sum(x, dims=1) ./ layer.factor
