export cnp_1d

"""
    cnp_1d(;
        dim_embedding::Integer,
        num_encoder_layers::Integer,
        num_decoder_layers::Integer
    )

# Arguments
- `dim_embedding::Integer`: Dimensionality of the embedding.
- `num_encoder_layers::Integer`: Number of layers in the encoder.
- `num_decoder_layers::Integer`: Number of layers in the decoder.

# Returns
- `Model`: Corresponding model.
"""
function cnp_1d(;
    dim_embedding::Integer,
    num_encoder_layers::Integer,
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
                        dim_out   =dim_embedding,
                        num_layers=num_encoder_layers
                    )
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
                num_layers=num_decoder_layers
            ),
            HeterogeneousGaussianLikelihood()
        )
    )
end
