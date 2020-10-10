export build_conv

function _compute_kernel_size(receptive_field, points_per_unit, num_layers)
    receptive_points = receptive_field * points_per_unit
    return ceil_odd(1 + (receptive_points - 1) / num_layers)
end

_compute_padding(kernel_size) = Integer(floor(kernel_size / 2))

_glorot_normal(dims...) = randn(Float32, dims...) .* sqrt(2f0 / sum(dims[1:2]) / dims[3])

_init_conv_fixed_bias(k, ch) =
    (_glorot_normal(k..., ch...), fill(1f-3, ch[2]))
_init_depthwiseconv_fixed_bias(k, ch) =
    (_glorot_normal(k..., div(ch[2], ch[1]), ch[1]), fill(1f-3, ch[2]))

_init_conv_random_bias(k, ch) =
    (_glorot_normal(k..., ch...), 1f-3 .* randn(Float32, ch[2]))
_init_depthwiseconv_random_bias(k, ch) =
    (_glorot_normal(k..., div(ch[2], ch[1]), ch[1]), 1f-3 .* randn(Float32, ch[2]))

_expand_kernel(n, ::Val{1}) = (n, 1)
_expand_kernel(n, ::Val{2}) = (n, n)

_expand_padding(n, ::Val{1}) = (n, 0)
_expand_padding(n, ::Val{2}) = (n, n)

"""
    build_conv(
        receptive_field::Float32,
        num_layers::Integer,
        num_channels::Integer;
        points_per_unit::Float32=30f0,
        multiple::Integer=1,
        num_in_channels::Integer=2,
        num_out_channels::Integer=2,
        dimensionality::Integer=1,
        init_conv::Function=_init_conv_fixed_bias,
        init_depthwiseconv::Function=_init_depthwiseconv_fixed_bias,
        act=x -> leakyrelu(x, 0.1f0)
    )

Build a CNN with a specified receptive field size.

# Arguments
- `receptive_field::Float32`: Width of the receptive field.
- `num_layers::Integer`: Number of layers of the CNN, excluding an initial
    and final pointwise convolutional layer to change the number of channels
    appropriately.
- `num_channels::Integer`: Number of channels of the CNN.

# Keywords
- `points_per_unit::Float32=30f0`: Points per unit for the discretisation. See
     `UniformDiscretisation1D`.
- `multiple::Integer=1`: Multiple for the discretisation. See `UniformDiscretisation1D`.
- `num_in_channels::Integer=2`: Number of input channels.
- `num_out_channels::Integer=2`: Number of output channels.
- `dimensionality::Integer=1`: Dimensionality of the filters.
- `init_conv::Function=_init_conv_fixed_bias`: Initialiser for dense convolutions.
- `init_depthwiseconv::Function=_init_depthwiseconv_fixed_bias`: Initialiser for depthwise
    separable convolutions.
- `act=x -> leakyrelu(x, 0.1f0)`: Activation function to use.

# Returns
- `BatchedConv`: CNN.
"""
function build_conv(
    receptive_field::Float32,
    num_layers::Integer,
    num_channels::Integer;
    points_per_unit::Float32=30f0,
    multiple::Integer=1,
    num_in_channels::Integer=2,
    num_out_channels::Integer=2,
    dimensionality::Integer=1,
    init_conv::Function=_init_conv_fixed_bias,
    init_depthwiseconv::Function=_init_depthwiseconv_fixed_bias,
    act=x -> leakyrelu(x, 0.1f0)
)
    # Appropriate expand the kernels to the right dimensionality.
    kernel = _expand_kernel(
        _compute_kernel_size(receptive_field, points_per_unit, num_layers),
        Val(dimensionality)
    )
    padding = _expand_padding(_compute_padding(kernel[1]), Val(dimensionality))

    # Build layers of the conv net.
    layers = Any[Conv(init_conv((1, 1), num_in_channels=>num_channels)..., act)]
    for i = 1:num_layers
        push!(layers, DepthwiseConv(
            init_depthwiseconv(kernel, num_channels=>num_channels)...,
            pad=padding
        ))
        push!(layers, Conv(
            init_conv((1, 1), num_channels=>num_channels)...,
            act
        ))
    end
    push!(layers, Conv(init_conv((1, 1), num_channels=>num_out_channels)...))

    return BatchedConv(
        Chain(layers...),
        points_per_unit,
        multiple,
        dimensionality
    )
end

"""
    struct BatchedConv

A batched CNN bundled with contextual information.

# Fields
- `conv`: Batched CNN.
- `points_per_unit::Float32`: Points per unit for the discretisation. See
     `UniformDiscretisation1D`.
- `multiple::Integer`: Multiple for the discretisation. See `UniformDiscretisation1D`.
- `dimensionality::Integer`: Dimensionality of the filters.
"""
struct BatchedConv
    conv
    points_per_unit::Float32
    multiple::Integer
    dimensionality::Integer
end

@Flux.functor BatchedConv

(layer::BatchedConv)(x) =  layer(x, Val(layer.dimensionality))

function (layer::BatchedConv)(x, dimensionality::Val{1})
    x, back = to_rank(3, x)        # Compress batch dimensions.
    x = with_dummy(layer.conv, x)  # Dummy dimension required!
    return back(x)                 # Uncompress batch dimensions
end

function (layer::BatchedConv)(x, dimensionality::Val{2})
    x, back = to_rank(4, x)  # Compress batch dimensions.
    x = layer.conv(x)        # No dummy dimension required!
    return back(x)           # Uncompress batch dimensions
end
