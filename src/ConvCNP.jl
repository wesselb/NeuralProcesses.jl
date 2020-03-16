export ConvCNP, convcnp_1d

"""
    ConvCNP

Convolutional CNP model.

# Fields
- `discretisation::Discretisation`: Discretisation for the encoding.
- `encoder::SetConv`: Encoder.
- `conv::Chain`: CNN that approximates rho.
- `decoder::SetConv`: Decoder.
"""
struct ConvCNP
    discretisation::Discretisation
    encoder::SetConv
    conv::Chain
    decoder::SetConv
end

@Flux.treelike ConvCNP

"""
    (model::ConvCNP)(
        x_context::AbstractArray{T, 3},
        y_context::AbstractArray{T, 3},
        x_target::AbstractArray{T, 3}
    ) where T<:Real

# Arguments
- `x_context::AbstractArray{T, 3}`: Locations of observed values of shape `(n, d, batch)`.
- `y_context::AbstractArray{T, 3}`: Observed values of shape `(n, channels, batch)`.
- `x_target::AbstractArray{T, 3}`: Locations of target set of shape `(m, d, batch)`.
"""
function (model::ConvCNP)(
    x_context::AbstractArray{T, 3},
    y_context::AbstractArray{T, 3},
    x_target::AbstractArray{T, 3}
) where T<:Real
    x_discretisation = model.discretisation(x_context, x_target)
    encoding = model.encoder(x_context, y_context, x_discretisation)
    latent = model.conv(encoding)
    if size(encoding, 1) != size(latent, 1)
        error("Conv net changed the discretisation size from $(size(encoding, 1)) to $(size(latent, 1)).")
    end
    return model.decoder(x_discretisation, latent, x_target)
end

"""
    convcnp_1d(arch::Architecture, margin::Real=0.1)

Construct a ConvCNP for one-dimensional data.

# Arguments
- `arch::Architecture`: CNN bundled with the points per units as constructed by
    `build_conv`.

# Keywords
- `margin::Real`: Margin for the discretisation. See `UniformDiscretisation1d`.
"""
function convcnp_1d(arch::Architecture, margin::Real=0.1)
    scale = 2 / arch.points_per_unit
    return ConvCNP(
        UniformDiscretisation1d(arch.points_per_unit, margin, arch.multiple),
        set_conv(1, scale; density=true),
        arch.conv,
        set_conv(2, scale; density=false)
    )
end
