export ConvNP, convnp_1d, sample_latent, encode, decode

"""
    ConvNP

Convolutional NP model.

# Fields
- `disc::Discretisation`: Discretisation for the encoding.
- `encoder::ConvCNP`: Encoder.
- `conv::Chain`: CNN that approximates ρ.
- `decoder::SetConv`: Decoder.
- `log_σ²`: Natural logarithm of observation noise variance.
"""
struct ConvNP
    disc::Discretisation
    encoder::ConvCNP
    conv::Chain
    decoder::SetConv
    log_σ²
end

@Flux.treelike ConvNP

"""
    (model::ConvNP)(
        x_context::AbstractArray,
        y_context::AbstractArray,
        x_target::AbstractArray
    )

# Arguments
- `x_context::AbstractArray`: Locations of observed values of shape `(n, d, batch)`.
- `y_context::AbstractArray`: Observed values of shape `(n, channels, batch)`.
- `x_target::AbstractArray`: Locations of target set of shape `(m, d, batch)`.
- `num_samples::Integer`: Number of samples.

# Returns
- `Tuple{AbstractArray, AbstractArray}`: Tuple containing means and variances.
"""

function (model::ConvNP)(
    x_context::AbstractArray,
    y_context::AbstractArray,
    x_target::AbstractArray,
    num_samples::Integer
)
    # Compute discretisation of the latent variable.
    x_latent = model.disc(x_context, x_target) |> gpu

    # Sample latent variable.
    samples = sample_latent(model, x_context, y_context, x_latent, num_samples)

    # Transform samples into predictions.
    return decode(model, x_latent, samples, x_target)
end

"""
    sample_latent(
        model::ConvNP,
        x_context::AbstractArray,
        y_context::AbstractArray,
        x_latent::AbstractArray,
        num_samples::Integer
    )

# Arguments
- `x_context::AbstractArray`: Locations of observed values of shape `(n, d, batch)`.
- `y_context::AbstractArray`: Observed values of shape `(n, y_channels, batch)`.
- `x_latent::AbstractArray`: Locations of latent variable of shape `(m, d, batch)`.
- `num_samples::Integer`: Number of samples.

# Returns
- `AbstractArray`: Samples of shape `(m, 1, latent_channels, batch, num_samples)`.
"""
function sample_latent(
    model::ConvNP,
    x_context::AbstractArray,
    y_context::AbstractArray,
    x_latent::AbstractArray,
    num_samples::Integer
)
    μ, σ² = encode(model, x_context, y_context, x_latent)
    return sample_latent(model, μ, σ², num_samples)
end

"""
    sample_latent(
        model::ConvNP,
        μ::AbstractArray,
        σ²::AbstractArray,
        num_samples::Integer
    )

# Arguments
- `μ::AbstractArray`: Means obtained from `encode`.
- `σ²::AbstractArray`: Variance obtains from `encode`.
- `num_samples::Integer`: Number of samples.

# Returns
- `AbstractArray`: Samples of shape `(m, 1, latent_channels, batch, num_samples)`.
"""
function sample_latent(
    model::ConvNP,
    μ::AbstractArray,
    σ²::AbstractArray,
    num_samples::Integer
)
    noise = randn(Float32, size(μ)..., num_samples) |> gpu
    return insert_dim(μ .+ sqrt.(σ²) .* noise, pos=2)
end

"""
    encode(
        model::ConvNP,
        x_context::AbstractArray,
        y_context::AbstractArray,
        x_latent::AbstractArray
    )

# Arguments
- `x_context::AbstractArray`: Locations of observed values of shape `(n, d, batch)`.
- `y_context::AbstractArray`: Observed values of shape `(n, channels, batch)`.
- `x_latent::AbstractArray`: Locations of latent variable of shape `(m, d, batch)`.

# Returns
- `Tuple{AbstractArray, AbstractArray}`: Tuple containing means and variances.
"""
function encode(
    model::ConvNP,
    x_context::AbstractArray,
    y_context::AbstractArray,
    x_latent::AbstractArray
)
    # Construct distribution over latent variable by calling the ConvCNP.
    return model.encoder(x_context, y_context, x_latent)
end

_repeat_samples(x, num_samples) = reshape(
    repeat(x, ntuple(_ -> 1, ndims(x))..., num_samples),
    size(x)[1:end - 1]...,
    :
)

"""
    decode(
        model::ConvNP,
        x_latent::AbstractArray,
        samples::AbstractArray,
        x_target::AbstractArray
    )

# Arguments
- `x_latent::AbstractArray`: Locations of latent variable of shape `(n, d, batch)`.
- `samples::AbstractArray`: Samples of shape `(n, 1, channels, batch, num_samples)`.
- `x_target::AbstractArray`: Locations of target set of shape `(m, d, batch)`.

# Returns
- `Tuple{AbstractArray, AbstractArray}`: Tuple containing means and variances.
"""
function decode(
    model::ConvNP,
    x_latent::AbstractArray,
    samples::AbstractArray,
    x_target::AbstractArray
)
    num_samples = size(samples, 5)

    # Merge samples into batches.
    samples = reshape(samples, size(samples)[1:3]..., :)

    # Apply CNN.
    channels = model.conv(samples)

    # Perform decoding.
    channels = decode(
        model.decoder,
        _repeat_samples(x_latent, num_samples),
        channels,
        _repeat_samples(x_target, num_samples)
    )

    # Separate samples from batches.
    channels = reshape(channels, size(channels)[1:3]..., :, num_samples)

    # Return predictive distribution.
    return channels, exp.(model.log_σ²)
end

"""
    convnp_1d(;
        receptive_field::Float32,
        num_layers::Integer,
        encoder_channels::Integer,
        decoder_channels::Integer,
        latent_channels::Integer,
        points_per_unit::Float32,
        margin::Float32=receptive_field,
        σ²::Float32=1f-2
    )

# Keywords
- `receptive_field::Float32`: Width of the receptive field.
- `num_layers::Integer`: Number of layers of the CNN, excluding an initial
    and final pointwise convolutional layer to change the number of channels
    appropriately.
- `encoder_channels::Integer`: Number of channels of the CNN of the encoder.
- `decoder_channels::Integer`: Number of channels of the CNN of the decoder.
- `latent_channels::Integer`: Number of channels of the latent variable.
- `margin::Float32=receptive_field`: Margin for the discretisation. See
    `UniformDiscretisation1d`.
- `σ²::Float32=1f-2`: Initialisation of the observation noise variance.
"""
function convnp_1d(;
    receptive_field::Float32,
    num_layers::Integer,
    encoder_channels::Integer,
    decoder_channels::Integer,
    latent_channels::Integer,
    points_per_unit::Float32,
    margin::Float32=receptive_field,
    σ²::Float32=1f-2
)
    # Build architecture for the encoder.
    arch_encoder = build_conv(
        receptive_field,
        num_layers,
        encoder_channels,
        points_per_unit=points_per_unit,
        dimensionality=1,
        in_channels=2,
        out_channels=2latent_channels
    )

    # Build architecture for the encoder.
    arch_decoder = build_conv(
        receptive_field,
        num_layers,
        decoder_channels,
        points_per_unit=points_per_unit,
        dimensionality=1,
        in_channels=latent_channels,
        out_channels=1
    )

    # Build encoder.
    scale = 2 / arch_encoder.points_per_unit
    encoder = ConvCNP(
        UniformDiscretisation1d(
            arch_encoder.points_per_unit,
            0f0,  # Do not use any margin. The ConvNP will account for this.
            arch_encoder.multiple
        ),
        set_conv(2, scale),  # Account for density channel.
        arch_encoder.conv,
        set_conv(2latent_channels, scale),
        _predict_gaussian_factorised
    )

    # Put model together.
    scale = 2 / arch_decoder.points_per_unit
    return ConvNP(
        UniformDiscretisation1d(
            arch_decoder.points_per_unit,
            margin,
            arch_decoder.multiple
        ),
        encoder,
        arch_decoder.conv,
        set_conv(1, scale),
        param([log(σ²)])
    )
end
