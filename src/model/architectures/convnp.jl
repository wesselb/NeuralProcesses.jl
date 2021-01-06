export convnp_1d

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
        noise_type::String="het",
        pooling_type::String="sum",
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
- `num_global_channels::Integer`: Number of channels of a global latent variable. Set to
    `0` to not use a global latent variable.
- `points_per_unit::Float32`: Points per unit for the discretisation. See
    `UniformDiscretisation1D`.
- `margin::Float32=receptive_field`: Margin for the discretisation. See
    `UniformDiscretisation1D`.
- `noise_type::String="het"`: Type of noise model. Must be "fixed", "amortised", or "het".
- `pooling_type::String="mean"`: Type of pooling. Must be "mean" or "sum".
- `σ::Float32=1f-2`: Initialisation of the fixed observation noise.
- `learn_σ::Bool=true`: Learn the fixed observation noise.

# Returns
- `Model`: Corresponding model.
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
    noise_type::String="het",
    pooling_type::String="mean",
    σ::Float32=1f-2,
    learn_σ::Bool=true
)
    dim_x = 1
    dim_y = 1
    scale = 2 / points_per_unit

    # Build encoder.
    encoder = Chain(
        set_conv(dim_y, scale; density=true),
        build_conv(
            receptive_field,
            num_encoder_layers,
            num_encoder_channels,
            points_per_unit =points_per_unit,
            dimensionality  =1,
            num_in_channels =dim_y + 1,  # Account for density channel.
            num_out_channels=2num_latent_channels + 2num_global_channels,
        ),
        HeterogeneousGaussianLikelihood()
    )
    if num_global_channels > 0
        # There is a global variable. Split it off and pool instead.
        if pooling_type == "mean"
            pooling = MeanPooling(layer_norm(1, 2num_global_channels, 1))
        elseif pooling_type == "sum"
            pooling = SumPooling(1000)  # Divide by `1000` to help initialisation.
        else
            error("Unknown pooling type \"" * pooling_type * "\".")
        end
        encoder = Chain(
            encoder[1:end - 1]...,  # Remove likelihood.
            Splitter(2num_global_channels),
            Parallel(
                HeterogeneousGaussianLikelihood(),
                Chain(
                    batched_mlp(
                        dim_in    =2num_global_channels,
                        dim_hidden=2num_global_channels,
                        dim_out   =2num_global_channels,
                        num_layers=3
                    ),
                    pooling,
                    batched_mlp(
                        dim_in    =2num_global_channels,
                        dim_hidden=2num_global_channels,
                        dim_out   =2num_global_channels,
                        num_layers=3
                    ),
                    HeterogeneousGaussianLikelihood()
                )
            )
        )
    end

    # Build noise model.
    num_noise_channels, noise = build_noise_model(
        # Need to perform a smoothing before applying the noise model.
        num_channels -> set_conv(num_channels, scale),
        dim_y       =dim_y,
        noise_type  =noise_type,
        pooling_type=pooling_type,
        σ           =σ,
        learn_σ     =learn_σ
    )

    return Model(
        FunctionalCoder(
            UniformDiscretisation1D(points_per_unit, margin),
            encoder
        ),
        Chain(
            Materialise(),
            build_conv(
                receptive_field,
                num_decoder_layers,
                num_decoder_channels,
                points_per_unit =points_per_unit,
                dimensionality  =1,
                num_in_channels =num_latent_channels + num_global_channels,
                num_out_channels=num_noise_channels
            ),
            noise
        )
    )
end
