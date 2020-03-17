export Architecture, build_conv

const Architecture = NamedTuple{
    (:conv, :points_per_unit, :multiple),
    Tuple{U, S, T}
} where U<:Chain where S<:Real where T<:Integer

function _compute_kernel_size(receptive_field, points_per_unit, num_layers)
    receptive_points = receptive_field * points_per_unit
    return ceil_odd(1 + (receptive_points - 1) / num_layers)
end

_compute_padding(kernel_size) = Integer(floor(kernel_size / 2))

"""
    build_conv(
        receptive_field,
        num_layers,
        num_channels;
        points_per_unit=64,
        multiple=1,
        in_channels=2,
        out_channels=1,
        dimensionality=1,
    )

Build a CNN with a specified receptive field size.

# Arguments
- `receptive_field::Real`: Width of the receptive field.
- `num_layers::Integer`: Number of layers of the CNN.
- `num_channels::Integer`: Number of channels of the CNN.

# Keywords
- `points_per_unit::Real=64`: Points per unit for the discretisation. See
     `UniformDiscretisation1d`.
- `multiple::Integer=1`: Multiple for the discretisation. See `UniformDiscretisation1d`.
- `in_channels::Integer=2`: Number of input channels.
- `out_channels::Integer=1`: Number of output channels.
- `dimensionality::Integer=1`: Dimensionality of the inputs. One corresponds to time series
    and two corresponds to images.

# Returns
- `Architecture`: Corresponding CNN bundled with the specified points per unit and margin.
"""
function build_conv(
    receptive_field::Real,
    num_layers::Integer,
    num_channels::Integer;
    points_per_unit::Real=64,
    multiple::Integer=1,
    in_channels::Integer=2,
    out_channels::Integer=2,
    dimensionality::Integer=1
)
    kernel_size = _compute_kernel_size(receptive_field, points_per_unit, num_layers)
    padding = _compute_padding(kernel_size)

    # Repeat the kernel size `dimensionality` many times to construct the
    # convolution kernel.
    kernel_pointwise = ntuple(_ -> 1, dimensionality)
    kernel = ntuple(_ -> kernel_size, dimensionality)

    # Build layers of the conv net.
    layers = []
    push!(layers, Conv(kernel_pointwise, in_channels=>num_channels, relu))
    for i = 1:(num_layers - 2)
        push!(layers, DepthwiseConv(
            kernel,
            num_channels=>num_channels,
            pad=padding,
            relu
        ))
    end
    push!(layers, Conv(kernel_pointwise, num_channels=>out_channels))

    return (
        conv=Chain(layers...),
        points_per_unit=points_per_unit,
        multiple=multiple
    )
end
