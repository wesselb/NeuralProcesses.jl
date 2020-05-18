export anp_1d

"""
    anp_1d(;
        dim_embedding::Integer,
        num_encoder_layers::Integer,
        num_encoder_heads::Integer,
        num_decoder_layers::Integer,
        σ::Float32=1f-2,
        learn_σ::Bool=true
    )

# Arguments
- `dim_embedding::Integer`: Dimensionality of the embedding.
- `num_encoder_layers::Integer`: Number of layers in the encoder.
- `num_encoder_heads::Integer`: Number of heads in the decoder.
- `num_decoder_layers::Integer`: Number of layers in the decoder.
- `σ::Float32=1f-2`: Initialisation of the observation noise.
- `learn_σ::Bool=true`: Learn the observation noise.

# Returns
- `NP`: Corresponding model.
"""
function anp_1d(;
    dim_embedding::Integer,
    num_encoder_layers::Integer,
    num_encoder_heads::Integer,
    num_decoder_layers::Integer,
    σ::Float32=1f-2,
    learn_σ::Bool=true
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
            dim_x=dim_x,
            dim_y=dim_y,
            dim_embedding=dim_embedding,
            num_heads=num_encoder_heads,
            num_encoder_layers=num_encoder_layers
        ),
        batched_mlp(
            dim_in    =2dim_embedding + dim_x,
            dim_hidden=dim_embedding,
            dim_out   =dim_y,
            num_layers=num_decoder_layers,
        ),
        learn_σ ? param([log(σ)]) : [log(σ)]
    )
end
