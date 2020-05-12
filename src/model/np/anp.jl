export anp_1d

"""
    anp_1d(;
        dim_embedding::Integer,
        num_encoder_layers::Integer,
        num_encoder_heads::Integer,
        num_decoder_layers::Integer,
        σ²::Float32=1f-4
    )

# Arguments
- `dim_embedding::Integer`: Dimensionality of the embedding.
- `num_encoder_layers::Integer`: Number of layers in the encoder.
- `num_encoder_heads::Integer`: Number of heads in the decoder.
- `num_decoder_layers::Integer`: Number of layers in the decoder.
- `σ²::Float32=1f-4`: Initialisation of the observation noise variance.

# Returns
- `NP`: Corresponding model.
"""
function anp_1d(;
    dim_embedding::Integer,
    num_encoder_layers::Integer,
    num_encoder_heads::Integer,
    num_decoder_layers::Integer,
    σ²::Float32=1f-3
)
    dim_x = 1
    dim_y = 1
    return NP(
        dim_embedding,
        attention(
            dim_x=dim_x,
            dim_y=dim_y,
            dim_embedding=dim_embedding,
            num_heads=num_encoder_heads,
            num_encoder_layers=num_encoder_layers
        ),
        NPEncoder(
            batched_mlp(
                dim_x + dim_y,
                2dim_embedding,
                2dim_embedding,
                num_encoder_layers
            )
        ),
        batched_mlp(
            2dim_embedding + dim_x,
            dim_embedding,
            dim_y,
            num_decoder_layers
        ),
        param([log(σ²)])
    )
end
