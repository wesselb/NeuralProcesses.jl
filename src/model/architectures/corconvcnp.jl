export corconvcnp_1d

"""
    corconvcnp_1d(;
        receptive_field_μ::Float32,
        receptive_field_Σ::Float32,
        num_layers::Integer,
        num_channels::Integer,
        points_per_unit_μ::Float32,
        points_per_unit_Σ::Float32,
        margin::Float32=receptive_field
    )

Construct a CorrelatedConvCNP for one-dimensional data.

# Keywords
- `receptive_field_μ::Float32`: Width of the receptive field for the mean.
- `receptive_field_Σ::Float32`: Width of the receptive field for the kernel.
- `num_layers::Integer`: Number of layers of the CNN, excluding an initial
    and final pointwise convolutional layer to change the number of channels
    appropriately.
- `num_channels::Integer`: Number of channels of the CNN.
- `points_per_unit_μ::Float32`: Points per unit for the discretisation for the mean. See
     `UniformDiscretisation1D`.
- `points_per_unit_Σ::Float32`: Points per unit for the discretisation for the kernel. See
     `UniformDiscretisation1D`.
- `margin::Float32=receptive_field`: Margin for the discretisation. See
    `UniformDiscretisation1D`.
"""
function corconvcnp_1d(;
    receptive_field_μ::Float32,
    receptive_field_Σ::Float32,
    num_layers::Integer,
    num_channels::Integer,
    points_per_unit_μ::Float32,
    points_per_unit_Σ::Float32,
    margin::Float32=receptive_field
)
    dim_x = 1
    dim_y = 1
    scale_μ = 2 / points_per_unit_μ
    scale_Σ = 2 / points_per_unit_Σ
    return Model(
        Parallel(
            FunctionalCoder(
                UniformDiscretisation1D(points_per_unit_μ, margin),
                Chain(
                    set_conv(dim_y, scale_μ; density=true),
                    DeterministicLikelihood()
                )
            ),
            FunctionalCoder(
                UniformDiscretisation1D(points_per_unit_Σ, margin),
                Chain(
                    set_conv(dim_y, scale_Σ; density=true, pd=true),
                    DeterministicLikelihood()
                )
            ),
        ),
        Chain(
            Parallel(
                Chain(
                    build_conv(
                        receptive_field_μ,
                        num_layers,
                        num_channels,
                        points_per_unit =points_per_unit_μ,
                        dimensionality  =1,
                        num_in_channels =dim_y + 1,  # Account for density channel.
                        num_out_channels=dim_y
                    ),
                    set_conv(dim_y, scale_μ)
                ),
                Chain(
                    build_conv(
                        receptive_field_Σ,
                        num_layers,
                        num_channels,
                        points_per_unit =points_per_unit_Σ,
                        dimensionality  =2,
                        # Account for density and identity channel.
                        num_in_channels =dim_y + 2,
                        num_out_channels=dim_y
                    ),
                    set_conv(dim_y, scale_Σ; pd=true)
                )
            ),
            CorrelatedGaussianLikelihood()
        )
    )
end
