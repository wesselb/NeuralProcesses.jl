export Architecture, build_conv

const Architecture = NamedTuple{
    (:conv, :points_per_unit, :multiple),
    Tuple{S, Float32, T}
} where S<:Chain where T<:Integer

function _compute_kernel_size(receptive_field, points_per_unit, num_layers)
    receptive_points = receptive_field * points_per_unit
    return ceil_odd(1 + (receptive_points - 1) / num_layers)
end

_compute_padding(kernel_size) = Integer(floor(kernel_size / 2))

_init_conv_fixed_bias(k, ch) =
    (Flux.glorot_normal(k..., ch...), fill(1f-3, ch[2]))
_init_depthwiseconv_fixed_bias(k, ch) =
    (Flux.glorot_normal(k..., div(ch[2], ch[1]), ch[1]), fill(1f-3, ch[2]))

_init_conv_random_bias(k, ch) =
    (Flux.glorot_normal(k..., ch...), 1f-3 .* randn(Float32, ch[2]))
_init_depthwiseconv_random_bias(k, ch) =
    (Flux.glorot_normal(k..., div(ch[2], ch[1]), ch[1]), 1f-3 .* randn(Float32, ch[2]))

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
        init_depthwiseconv::Function=_init_depthwiseconv_fixed_bias
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
     `UniformDiscretisation1d`.
- `multiple::Integer=1`: Multiple for the discretisation. See `UniformDiscretisation1d`.
- `num_in_channels::Integer=2`: Number of input channels.
- `num_out_channels::Integer=2`: Number of output channels.
- `dimensionality::Integer=1`: Dimensionality of the filters.
- `init_conv::Function=_init_conv_fixed_bias`: Initialiser for dense convolutions.
- `init_depthwiseconv::Function=_init_depthwiseconv_fixed_bias`: Initialiser for depthwise
    separable convolutions.

# Returns
- `Architecture`: Corresponding CNN bundled with the specified points per unit and margin.
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
    init_depthwiseconv::Function=_init_depthwiseconv_fixed_bias
)
    # Appropriate expand the kernels to the right dimensionality.
    kernel = _expand_kernel(
        _compute_kernel_size(receptive_field, points_per_unit, num_layers),
        Val(dimensionality)
    )
    padding = _expand_padding(_compute_padding(kernel[1]), Val(dimensionality))

    act(x) = leakyrelu(x, 0.1f0)

    # Build layers of the conv net.
    layers = Any[Conv(Flux.param.(init_conv((1, 1), num_in_channels=>num_channels))..., act)]
    for i = 1:num_layers
        push!(layers, DepthwiseConv(
            Flux.param.(init_depthwiseconv(kernel, num_channels=>num_channels))...,
            pad=padding
        ))
        push!(layers, Conv(
            Flux.param.(init_conv((1, 1), num_channels=>num_channels))...,
            act
        ))
    end
    push!(layers, Conv(Flux.param.(init_conv((1, 1), num_channels=>num_out_channels))...))

    return (
        conv=Chain(layers...),
        points_per_unit=points_per_unit,
        multiple=multiple
    )
end
