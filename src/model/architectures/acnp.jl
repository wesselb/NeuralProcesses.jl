export acnp_1d

"""
    acnp_1d(;
        dim_embedding::Integer,
        num_encoder_layers::Integer,
        num_encoder_heads::Integer,
        num_decoder_layers::Integer
    )

# Arguments
- `dim_embedding::Integer`: Dimensionality of the embedding.
- `num_encoder_layers::Integer`: Number of layers in the encoder.
- `num_encoder_heads::Integer`: Number of heads in the encoder.
- `num_decoder_layers::Integer`: Number of layers in the decoder.

# Returns
- `Model`: Corresponding model.
"""
function acnp_1d(;
    dim_embedding::Integer,
    num_encoder_layers::Integer,
    num_encoder_heads::Integer,
    num_decoder_layers::Integer
)
    dim_x = 1
    dim_y = 1
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
            )
        ),
        Chain(
            Materialise(),
            batched_mlp(
                dim_in    =dim_x + dim_embedding,
                dim_hidden=dim_embedding,
                dim_out   =2dim_y,
                num_layers=num_decoder_layers,
            ),
            HeterogeneousGaussianLikelihood()
        )
    )
end
