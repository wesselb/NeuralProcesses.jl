export ConvNP, convnp_1d, loglik, elbo, predict

"""
    ConvNP

Convolutional NP model.

# Fields
- `disc::Discretisation`: Discretisation for the encoding.
- `encoder::ConvCNP`: Encoder.
- `conv::Chain`: CNN that approximates ρ.
- `decoder::SetConv`: Decoder.
- `log_σ`: Natural logarithm of observation noise.
"""
struct ConvNP <: AbstractNP
    disc::Discretisation
    encoder::ConvCNP
    conv::Chain
    decoder::SetConv
    log_σ
end

@Flux.treelike ConvNP

encoding_locations(model::ConvNP, xc, xt) = model.disc(xc, xt) |> gpu

encode_lat(model::ConvNP, xc, yc, xz) = model.encoder(xc, yc, xz)

empty_lat_encoding(model::ConvNP, xz) = model.encoder(nothing, nothing, xz)

encode_det(model::ConvNP, xc, yc, xz) = nothing

empty_det_encoding(model::ConvNP, xz) = nothing

function decode(model::ConvNP, xz, z, r::Nothing, xt)
    _, _, num_batches, num_samples = size(z)

    # Merge samples into batches.
    z = reshape(z, size(z)[1:2]..., num_batches * num_samples)

    # Apply CNN.
    channels = with_dummy(model.conv, z)

    # Perform decoding.
    channels = decode(
        model.decoder,
        _repeat_samples(xz, num_samples),
        channels,
        _repeat_samples(xt, num_samples)
    )

    # Separate samples from batches.
    channels = reshape(channels, size(channels)[1:2]..., num_batches, num_samples)

    return channels
end

decode(model::ConvNP, xz, z::Tuple, r::Nothing, xt) =
    decode(model, xz, repeat_cat(z..., dims=2), r, xt)

_repeat_samples(x, num_samples) = reshape(
    repeat(x, ntuple(_ -> 1, ndims(x))..., num_samples),
    size(x)[1:end - 1]...,
    size(x)[end] * num_samples
)

"""
    convnp_1d(;
        receptive_field::Float32,
        num_encoder_layers::Integer,
        num_decoder_layers::Integer,
        num_encoder_channels::Integer,
        num_decoder_channels::Integer,
        num_latent_channels::Integer,
        points_per_unit::Float32,
        margin::Float32=receptive_field,
        σ::Float32=1f-2,
        learn_σ::Bool=true,
        global_variable::Bool=true
    )

# Keywords
- `receptive_field::Float32`: Width of the receptive field.
- `num_layers::Integer`: Number of layers of the CNN, excluding an initial
    and final pointwise convolutional layer to change the number of channels
    appropriately.
- `num_encoder_layers::Integer`: Number of layers of the CNN of the encoder.
- `num_decoder_layers::Integer`: Number of layers of the CNN of the decoder.
- `num_encoder_channels::Integer`: Number of channels of the CNN of the encoder.
- `num_decoder_channels::Integer`: Number of channels of the CNN of the decoder.
- `num_latent_channels::Integer`: Number of channels of the latent variable.
- `margin::Float32=receptive_field`: Margin for the discretisation. See
    `UniformDiscretisation1d`.
- `σ::Float32=1f-2`: Initialisation of the observation noise.
- `learn_σ::Bool=true`: Learn the observation noise.
- `global_variable::Bool=true`: Also use a global latent variable.

# Returns
- `ConvNP`: Corresponding model.
"""
function convnp_1d(;
    receptive_field::Float32,
    num_encoder_layers::Integer,
    num_decoder_layers::Integer,
    num_encoder_channels::Integer,
    num_decoder_channels::Integer,
    num_latent_channels::Integer,
    points_per_unit::Float32,
    margin::Float32=receptive_field,
    σ::Float32=1f-2,
    learn_σ::Bool=true,
    global_variable::Bool=true
)
    # Build architecture for the encoder.
    arch_encoder = build_conv(
        receptive_field,
        num_encoder_layers,
        num_encoder_channels,
        points_per_unit=points_per_unit,
        dimensionality=1,
        num_in_channels=2,  # Account for density channel.
        num_out_channels=2num_latent_channels  # Outputs means and standard deviations.
    )

    # Build architecture for the encoder.
    arch_decoder = build_conv(
        receptive_field,
        num_decoder_layers,
        num_decoder_channels,
        points_per_unit=points_per_unit,
        dimensionality=1,
        num_in_channels=num_latent_channels,
        num_out_channels=1
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
        set_conv(2num_latent_channels, scale),
        # Insert the global variable by performing a mean pooling of half the latent
        # channels here.
        global_variable ? x -> split_μ_σ.(_split_mean_pool(x)) : split_μ_σ
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
        learn_σ ? param([log(σ)]) : [log(σ)]
    )
end

function _split_mean_pool(x)
    # Perform mean pooling on the first half of the channels.
    x₁, x₂ = split(x, 2)
    return mean(x₁, dims=1), x₂
end
