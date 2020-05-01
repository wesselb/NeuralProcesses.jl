export ConvCNP, convcnp_1d

"""
    ConvCNP

Convolutional CNP model with a factorised Gaussian predictive distribution.

# Fields
- `disc::Discretisation`: Discretisation for the encoding.
- `encoder::SetConv`: Encoder.
- `conv::Chain`: CNN that approximates ρ.
- `decoder::SetConv`: Decoder.
- `predict`: Function that transforms the decoding into a predictive distribution.
"""
struct ConvCNP
    disc::Discretisation
    encoder::SetConv
    conv::Chain
    decoder::SetConv
    predict
end

@Flux.treelike ConvCNP

"""
    (model::ConvCNP)(
        x_context::AbstractArray,
        y_context::AbstractArray,
        x_target::AbstractArray
    )

# Arguments
- `x_context::AbstractArray`: Locations of observed values of shape `(n, d, batch)`.
- `y_context::AbstractArray`: Observed values of shape `(n, channels, batch)`.
- `x_target::AbstractArray`: Locations of target set of shape `(m, d, batch)`.

# Returns
- `Tuple{AbstractArray, AbstractArray}`: Tuple containing means and variances.
"""
function (model::ConvCNP)(
    x_context::AbstractArray,
    y_context::AbstractArray,
    x_target::AbstractArray
)
    # Compute discretisation of the functional embedding.
    x_latent = model.disc(x_context, x_target) |> gpu

    if size(x_context, 1) > 0
        # The context set is non-empty. Compute encoding as usual.
        channels = encode(model.encoder, x_context, y_context, x_latent)
    else
        # The context set is empty. Set to empty encodings.
        channels = empty_encoding(model.encoder, x_latent)
    end

    # Apply CNN.
    channels = model.conv(channels)

    # Perform decoding.
    channels = decode(model.decoder, x_latent, channels, x_target)

    # Return predictive distribution.
    return model.predict(channels)
end

function _predict_gaussian_factorised(channels)
    size(channels, 2) == 1 || error("Channels are not one-dimensional.")
    mod(size(channels, 3), 2) == 0 || error("Number of channels must be even.")

    # Half of the channels are used to determine the mean, and the other half are used to
    # determine the standard deviation.
    i_split = div(size(channels, 3), 2)
    μ = channels[:, 1, 1:i_split, :]
    σ² = NNlib.softplus.(channels[:, 1, i_split + 1:end, :])
    return μ, σ²
end


"""
    convcnp_1d(;
        receptive_field::Float32,
        num_layers::Integer,
        num_channels::Integer,
        points_per_unit::Float32,
        margin::Float32=receptive_field
    )

Construct a ConvCNP for one-dimensional data.

# Keywords
- `receptive_field::Float32`: Width of the receptive field.
- `num_layers::Integer`: Number of layers of the CNN, excluding an initial
    and final pointwise convolutional layer to change the number of channels
    appropriately.
- `num_channels::Integer`: Number of channels of the CNN.
- `margin::Float32=receptive_field`: Margin for the discretisation. See
    `UniformDiscretisation1d`.
"""
function convcnp_1d(;
    receptive_field::Float32,
    num_layers::Integer,
    num_channels::Integer,
    points_per_unit::Float32,
    margin::Float32=receptive_field
)
    # Build architecture.
    arch = build_conv(
        receptive_field,
        num_layers,
        num_channels,
        points_per_unit=points_per_unit,
        dimensionality=1,
        in_channels=2,
        out_channels=2
    )

    scale = 2 / arch.points_per_unit
    return ConvCNP(
        UniformDiscretisation1d(arch.points_per_unit, margin, arch.multiple),
        set_conv(2, scale),  # Account for density channel.
        arch.conv,
        set_conv(2, scale),
        _predict_gaussian_factorised
    )
end
