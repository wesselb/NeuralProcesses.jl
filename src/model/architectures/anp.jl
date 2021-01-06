export anp_1d

"""
    anp_1d(;
        dim_embedding::Integer,
        num_encoder_layers::Integer,
        num_encoder_heads::Integer,
        num_decoder_layers::Integer,
        noise_type::String="het",
        pooling_type::String="mean",
        σ::Float32=1f-2,
        learn_σ::Bool=true
    )

# Arguments
- `dim_embedding::Integer`: Dimensionality of the embedding.
- `num_encoder_layers::Integer`: Number of layers in the encoder.
- `num_encoder_heads::Integer`: Number of heads in the encoder.
- `num_decoder_layers::Integer`: Number of layers in the decoder.
- `noise_type::String="het"`: Type of noise model. Must be "fixed", "amortised", or "het".
- `pooling_type::String="mean"`: Type of pooling. Must be "mean" or "sum".
- `σ::Float32=1f-2`: Initialisation of the fixed observation noise.
- `learn_σ::Bool=true`: Learn the fixed observation noise.

# Returns
- `Model`: Corresponding model.
"""
function anp_1d(;
    dim_embedding::Integer,
    num_encoder_layers::Integer,
    num_encoder_heads::Integer,
    num_decoder_layers::Integer,
    noise_type::String="het",
    pooling_type::String="mean",
    σ::Float32=1f-2,
    learn_σ::Bool=true
)
    dim_x = 1
    dim_y = 1
    num_noise_channels, noise = build_noise_model(
        dim_y       =dim_y,
        noise_type  =noise_type,
        pooling_type=pooling_type,
        σ           =σ,
        learn_σ     =learn_σ
    )
    return Model(
        Parallel(
            Chain(
                InputsCoder(),
                DeterministicLikelihood()
            ),
            Chain(
                attention(
                    dim_x             =dim_x,
                    dim_y             =dim_y,
                    dim_embedding     =dim_embedding,
                    num_heads         =num_encoder_heads,
                    num_encoder_layers=num_encoder_layers
                ),
                DeterministicLikelihood()
            ),
            Chain(
                MLPCoder(
                    batched_mlp(
                        dim_in    =dim_x + dim_y,
                        dim_hidden=dim_embedding,
                        dim_out   =dim_embedding,
                        num_layers=num_encoder_layers
                    ),
                    batched_mlp(
                        dim_in    =dim_embedding,
                        dim_hidden=dim_embedding,
                        dim_out   =2dim_embedding,
                        num_layers=num_encoder_layers
                    )
                ),
                HeterogeneousGaussianLikelihood()
            )
        ),
        Chain(
            Materialise(),
            batched_mlp(
                dim_in    =dim_x + 2dim_embedding,
                dim_hidden=dim_embedding,
                dim_out   =num_noise_channels,
                num_layers=num_decoder_layers
            ),
            noise
        )
    )
end
