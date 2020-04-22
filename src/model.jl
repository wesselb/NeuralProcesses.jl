export ConvCNP, convcnp_1d, CorrelatedConvCNP, convcnp_1d_correlated

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

function _check_conv_output(encoding, latent)
    shape_encoding = size(encoding)[[1, 2]]
    shape_latent = size(latent)[[1, 2]]
    if shape_encoding != shape_latent
        error("Conv net changed the shape from $(shape_encoding) to $(shape_latent).")
    end
end

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
    x_disc = gpu(model.disc(x_context, x_target))

    if size(x_context, 1) > 0
        # The context set is non-empty. Compute encoding as usual.
        encoding = encode(model.encoder, x_context, y_context, x_disc)
    else
        # The context set is empty. Set to empty encodings.
        encoding = empty_encoding(model.encoder, y_context, x_disc)
    end

    # Apply CNN.
    latent = model.conv(encoding)
    _check_conv_output(encoding, latent)

    # Perform decoding.
    channels = decode(model.decoder, x_disc, latent, x_target)

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
    convcnp_1d(arch::Architecture, margin::Float32=0.1f0)

Construct a ConvCNP for one-dimensional data.

# Arguments
- `arch::Architecture`: CNN bundled with the points per units as constructed by
    `build_conv`.

# Keywords
- `margin::Float32=0.1f0`: Margin for the discretisation. See `UniformDiscretisation1d`.
"""
function convcnp_1d(arch::Architecture; margin::Float32=0.1f0)
    scale = 2 / arch.points_per_unit
    return ConvCNP(
        UniformDiscretisation1d(arch.points_per_unit, margin, arch.multiple),
        set_conv(2, scale),  # Account for density channel.
        arch.conv,
        set_conv(2, scale),
        _predict_gaussian_factorised
    )
end

"""
    CorrelatedConvCNP

Convolutional CNP model with a correlated Gaussian predictive distribution.

# Fields
- `μ_disc::Discretisation`: Discretisation for the encoding of the mean.
- `Σ_disc::Discretisation`: Discretisation for the encoding of the covariance.
- `μ_encoder::SetConv`: Encoder for the mean.
- `Σ_encoder::SetConv`: Encoder for the covariance.
- `μ_conv::Chain`: CNN that approximates ρ for the mean.
- `Σ_conv::Chain`: CNN that approximates ρ for the covariance.
- `μ_decoder::SetConv`: Decoder for the mean.
- `Σ_decoder::SetConv`: Decoder for the covariance.
- `predict`: Function that transforms the decodings into a predictive distribution.
"""
struct CorrelatedConvCNP
    μ_disc::Discretisation
    Σ_disc::Discretisation
    μ_encoder::SetConv
    Σ_encoder::SetConv
    μ_conv::Chain
    Σ_conv::Chain
    μ_decoder::SetConv
    Σ_decoder::SetConv
    predict
end

@Flux.treelike CorrelatedConvCNP

"""
    (model::CorrelatedConvCNP)(
        x_context::AbstractArray,
        y_context::AbstractArray,
        x_target::AbstractArray
    )

# Arguments
- `x_context::AbstractArray`: Locations of observed values of shape `(n, d, batch)`.
- `y_context::AbstractArray`: Observed values of shape `(n, channels, batch)`.
- `x_target::AbstractArray`: Locations of target set of shape `(m, d, batch)`.

# Returns
- `Tuple{AbstractArray, AbstractArray}`: Tuple containing means and covariances.
"""
function (model::CorrelatedConvCNP)(
    x_context::AbstractArray,
    y_context::AbstractArray,
    x_target::AbstractArray
)
    # Compute discretisation of the functional embedding.
    μ_disc = gpu(model.μ_disc(x_context, x_target))
    Σ_disc = gpu(model.Σ_disc(x_context, x_target))

    if size(x_context, 1) > 0
        # The context set is non-empty. Compute encodings as usual.
        μ_encoding = encode(   model.μ_encoder, x_context, y_context, μ_disc)
        Σ_encoding = encode_pd(model.Σ_encoder, x_context, y_context, Σ_disc)
    else
        # The context set is empty. Set to empty encodings.
        μ_encoding = empty_encoding(   model.μ_encoder, y_context, μ_disc)
        Σ_encoding = empty_encoding_pd(model.Σ_encoder, y_context, Σ_disc)
    end

    # Apply the CNNs.
    μ_latent = model.μ_conv(μ_encoding)
    _check_conv_output(μ_encoding, μ_latent)
    Σ_latent = model.Σ_conv(Σ_encoding)
    _check_conv_output(Σ_encoding, Σ_latent)

    # Perform decoding.
    μ_channels = decode(   model.μ_decoder, μ_disc, μ_latent, x_target)
    Σ_channels = decode_pd(model.Σ_decoder, Σ_disc, Σ_latent, x_target)

    # Return predictive distribution.
    return model.predict(μ_channels, Σ_channels)
end

"""
    convcnp_1d_correlated(arch::Architecture, margin::Float32=0.1f0)

Construct a correlated ConvCNP for one-dimensional data.

# Arguments
- `arch::Architecture`: CNN bundled with the points per units as constructed by
    `build_conv`.

# Keywords
- `margin::Float32=0.1f0`: Margin for the discretisation. See `UniformDiscretisation1d`.
"""
function convcnp_1d_correlated(
    μ_arch::Architecture,
    Σ_arch::Architecture;
    margin::Float32=0.1f0
)
    μ_scale = 2 / μ_arch.points_per_unit
    Σ_scale = 2 / Σ_arch.points_per_unit
    return CorrelatedConvCNP(
        UniformDiscretisation1d(μ_arch.points_per_unit, margin, μ_arch.multiple),
        UniformDiscretisation1d(Σ_arch.points_per_unit, margin, Σ_arch.multiple),
        set_conv(2, μ_scale),  # Account for density channel.
        set_conv(2, Σ_scale),  # Account for density channel.
        μ_arch.conv,
        Σ_arch.conv,
        set_conv(1, μ_scale),
        set_conv(1, Σ_scale),
        _predict_gaussian_correlated
    )
end

function _predict_gaussian_correlated(μ_channels, Σ_channels)
    size(μ_channels, 2) == 1 || error("Mean is not one-dimensional.")
    size(μ_channels, 3) == 1 || error("More than one channel for the mean.")
    size(Σ_channels, 3) == 1 || error("More than one channel for the covariance.")
    return μ_channels[:, 1, 1, :], Σ_channels[:, :, 1, :]
end
