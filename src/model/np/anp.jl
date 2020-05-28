export anp_1d

"""
    anp_1d(;
        dim_embedding::Integer,
        num_encoder_layers::Integer,
        num_encoder_heads::Integer,
        num_decoder_layers::Integer,
        num_σ_channels::Integer,
        σ::Float32=1f-2,
        learn_σ::Bool=true,
        pooling_type::String="sum"
    )

# Arguments
- `dim_embedding::Integer`: Dimensionality of the embedding.
- `num_encoder_layers::Integer`: Number of layers in the encoder.
- `num_encoder_heads::Integer`: Number of heads in the decoder.
- `num_decoder_layers::Integer`: Number of layers in the decoder.
- `num_σ_channels::Integer`: Learn the observation noise through amortisation. Set to
    `0` to use a constant noise. If set to a value bigger than `0`, the values for `σ` and
    `learn_σ` are ignored.
- `σ::Float32=1f-2`: Initialisation of the observation noise.
- `learn_σ::Bool=true`: Learn the observation noise.
- `pooling_type::String="sum"`: Type of pooling. Must be "sum" or "mean".

# Returns
- `NP`: Corresponding model.
"""
function anp_1d(;
    dim_embedding::Integer,
    num_encoder_layers::Integer,
    num_encoder_heads::Integer,
    num_decoder_layers::Integer,
    num_σ_channels::Integer,
    σ::Float32=1f-2,
    learn_σ::Bool=true,
    pooling_type::String="sum"
)
    dim_x = 1
    dim_y = 1
    return NP(
        NPEncoder(
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
                num_layers=2
            )
        ),
        attention(
            dim_x             =dim_x,
            dim_y             =dim_y,
            dim_embedding     =dim_embedding,
            num_heads         =num_encoder_heads,
            num_encoder_layers=num_encoder_layers
        ),
        batched_mlp(
            dim_in    =2dim_embedding + dim_x,
            dim_hidden=dim_embedding,
            dim_out   =num_σ_channels + dim_y,
            num_layers=num_decoder_layers,
        ),
        _np_build_noise_model(
            num_σ_channels=num_σ_channels,
            σ             =σ,
            learn_σ       =learn_σ,
            pooling_type  =pooling_type
        )
    )
end
