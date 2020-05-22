export ConvNP, convnp_1d, loglik, elbo, predict

"""
    ConvNP

Convolutional NP model.

# Fields
- `disc::Discretisation`: Discretisation for the encoding.
- `encoder_setconv::ConvCNP`: Set convolution for the encoder.
- `encoder_conv::Chain`: Encoder CNN.
- `encoder_predict`: Function that transforms the output of the encoder into a distribution.
- `decoder_conv::Chain`: Decoder CNN.
- `decoder_setconv::SetConv`: Set convolution for the decoder.
- `log_σ`: Natural logarithm of observation noise.
"""
struct ConvNP <: AbstractNP
    disc::Discretisation
    encoder_setconv::SetConv
    encoder_conv::Chain
    encoder_predict
    decoder_conv::Chain
    decoder_setconv::SetConv
    log_σ
end

@Flux.treelike ConvNP

encoding_locations(model::ConvNP, xc::AA, xt::AA) = model.disc(xc, xt) |> gpu

function encode_lat(model::ConvNP, xc::AA, yc::AA, xz::AA)
    # The context set is non-empty.
    channels = encode(model.encoder_setconv, xc, yc, xz)

    # Apply CNN.
    channels = with_dummy(model.encoder_conv, channels)

    # Return distribution.
    return model.encoder_predict(channels)
end

function empty_lat_encoding(model::ConvNP, xz::AA)
    # The context set is empty, so get the encoding for the empty set.
    channels = empty_encoding(model.encoder_setconv, xz)

    # Still apply the CNN!
    channels = with_dummy(model.encoder_conv, channels)

    # Return distribution.
    return model.encoder_predict(channels)
end

encode_det(model::ConvNP, xc::AA, yc::AA, xz::AA) = nothing

empty_det_encoding(model::ConvNP, xz::AA) = nothing

function decode(model::ConvNP, xz::AA, z::AA, r::Nothing, xt::AA)
    _, _, num_batches, num_samples = size(z)

    # Merge samples into batches.
    z = reshape(z, size(z)[1:2]..., num_batches * num_samples)

    # Apply CNN.
    channels = with_dummy(model.decoder_conv, z)

    # Perform decoding.
    channels = decode(
        model.decoder_setconv,
        _repeat_samples(xz, num_samples),
        channels,
        _repeat_samples(xt, num_samples)
    )

    # Separate samples from batches.
    channels = reshape(channels, size(channels)[1:2]..., num_batches, num_samples)

    return channels
end

decode(model::ConvNP, xz::AA, z::Tuple, r::Nothing, xt::AA) =
    decode(model, xz, repeat_cat(z..., dims=2), r, xt)

_repeat_samples(x, num_samples) = reshape(
    repeat_gpu(x, ntuple(_ -> 1, ndims(x))..., num_samples),
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
        num_global_channels::Integer,
        points_per_unit::Float32,
        margin::Float32=receptive_field,
        σ::Float32=1f-2,
        learn_σ::Bool=true
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
- `num_global_channels::Integer`: Number of channels of the global latent variable. Set
    to `0` to not use a global latent variable.
- `margin::Float32=receptive_field`: Margin for the discretisation. See
    `UniformDiscretisation1d`.
- `σ::Float32=1f-2`: Initialisation of the observation noise.
- `learn_σ::Bool=true`: Learn the observation noise.

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
    num_global_channels::Integer,
    points_per_unit::Float32,
    margin::Float32=receptive_field,
    σ::Float32=1f-2,
    learn_σ::Bool=true
)
    # Build architecture for the encoder.
    arch_encoder = build_conv(
        receptive_field,
        num_encoder_layers,
        num_encoder_channels,
        points_per_unit=points_per_unit,
        dimensionality=1,
        num_in_channels=2,  # Account for density channel.
        # Outputs means and standard deviations.
        num_out_channels=2(num_latent_channels + num_global_channels)
    )

    # Build architecture for the encoder.
    arch_decoder = build_conv(
        receptive_field,
        num_decoder_layers,
        num_decoder_channels,
        points_per_unit=points_per_unit,
        dimensionality=1,
        num_in_channels=num_latent_channels + num_global_channels,
        num_out_channels=1
    )

    # If we are using a global variable, split it off of the end of the encoder.
    if num_global_channels > 0
        encoder_predict = SplitGlobalVariable(
            2num_global_channels,
            layer_norm(1, 2num_global_channels, 1),
            batched_mlp(
                dim_in    =2num_global_channels,
                dim_hidden=2num_global_channels,
                dim_out   =2num_global_channels,
                num_layers=3
            ),
            split_μ_σ
        )
    else
        encoder_predict = split_μ_σ
    end

    # Put model together.
    scale = 2 / arch_decoder.points_per_unit
    return ConvNP(
        UniformDiscretisation1d(
            arch_encoder.points_per_unit,
            margin,
            arch_encoder.multiple
        ),
        set_conv(2, scale),  # Account for density channel.
        arch_encoder.conv,
        encoder_predict,
        arch_decoder.conv,
        set_conv(1, scale),
        learn_σ ? param([log(σ)]) : [log(σ)]
    )
end

"""
    struct SplitGlobalVariable

# Fields
- `num_global_channels::Integer`: Number of channels to use for the global variable.
- `ln`: Layer normalisation to correct the pooling.
- `ff`: Feed-forward net to allow the magnitude to change after the layer norm.
- `predict`: Function that transforms the output into a distribution.
"""
struct SplitGlobalVariable
    num_global_channels::Integer
    ln
    ff
    predict
end

@Flux.treelike SplitGlobalVariable

"""
    (layer::SplitGlobalVariable)(x::AA)

# Arguments
- `x::AA`: Output of the encoder. Half of the channels will be used to construct the global
    variable.

# Returns
- `Tuple`: Two-tuple containing the distributions for the global and equivariant variable.
"""
function (layer::SplitGlobalVariable)(x::AA)

    # Split channels.
    x₁ = x[:, 1:layer.num_global_channels, :]
    x₂ = x[:, layer.num_global_channels + 1:end, :]

    # Mean-pool over data points to make global channels. Also normalise to not depend
    # on the size of the discretisation.
    x₁ = mean(x₁, dims=1)
    x₁ = layer.ln(x₁)

    # Apply a FF network to allow the magnitude to change.
    x₁ = layer.ff(x₁)

    return layer.predict(x₁), layer.predict(x₂)
end
