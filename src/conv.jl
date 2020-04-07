export Architecture, build_conv_1d

const Architecture = NamedTuple{
    (:conv, :points_per_unit, :multiple),
    Tuple{S, Float32, T}
} where S<:Chain where T<:Integer

function _compute_kernel_size(receptive_field, points_per_unit, num_layers)
    receptive_points = receptive_field * points_per_unit
    return ceil_odd(1 + (receptive_points - 1) / num_layers)
end

_compute_padding(kernel_size) = Integer(floor(kernel_size / 2))

_init_conv_fixed_bias(k, ch) = (
    Flux.param(Flux.glorot_normal(k..., ch...)),
    Flux.param(fill(1f-3, ch[2]))
)

_init_depthwiseconv_fixed_bias(k, ch) = (
    Flux.param(Flux.glorot_normal(k..., div(ch[2], ch[1]), ch[1])),
    Flux.param(fill(1f-3, ch[2]))
)

"""
    build_conv(
        receptive_field::Float32,
        num_layers::Integer,
        num_channels::Integer;
        points_per_unit::Float32=30f0,
        multiple::Integer=1,
        in_channels::Integer=2,
        out_channels::Integer=2,
        init_conv::Function=_init_conv_fixed_bias,
        init_depthwiseconv::Function=_init_depthwiseconv_fixed_bias
    )

Build a 1D CNN with a specified receptive field size.

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
- `in_channels::Integer=2`: Number of input channels.
- `out_channels::Integer=2`: Number of output channels.
- `init_conv::Function=_init_conv_fixed_bias`: Initialiser for dense convolutions.
- `init_depthwiseconv::Function=_init_depthwiseconv_fixed_bias`: Initialiser for depthwise
    separable convolutions.

# Returns
- `Architecture`: Corresponding CNN bundled with the specified points per unit and margin.
"""
function build_conv_1d(
    receptive_field::Float32,
    num_layers::Integer,
    num_channels::Integer;
    points_per_unit::Float32=30f0,
    multiple::Integer=1,
    in_channels::Integer=2,
    out_channels::Integer=2,
    init_conv::Function=_init_conv_fixed_bias,
    init_depthwiseconv::Function=_init_depthwiseconv_fixed_bias
)
    # We use two-dimensional kernels: CUDNN does not support 1D convolutions.
    kernel = (_compute_kernel_size(receptive_field, points_per_unit, num_layers), 1)
    padding = (_compute_padding(kernel[1]), 0)

    act(x) = leakyrelu(x, 0.1f0)

    # Build layers of the conv net.
    layers = []
    push!(layers, Conv(init_conv((1, 1), in_channels=>num_channels)..., act))
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
    push!(layers, Conv(init_conv((1, 1), num_channels=>out_channels)...))

    return (
        conv=Chain(layers...),
        points_per_unit=points_per_unit,
        multiple=multiple
    )
end
