export ConvCNP, convcnp_1d_factorised, convcnp_1d_lowrank

"""
    ConvCNP

Convolutional CNP model.

# Fields
- `discretisation::Discretisation`: Discretisation for the encoding.
- `encoder::SetConv`: Encoder.
- `conv::Chain`: CNN that approximates rho.
- `decoder::SetConv`: Decoder.
- `predict::Function`: Function that transforms the decoding into a predictive distribution.
"""
struct ConvCNP
    discretisation::Discretisation
    encoder::SetConv
    conv::Chain
    decoder::SetConv
    predict
end

@Flux.treelike ConvCNP

"""
    (model::ConvCNP)(
        x_context::AbstractArray,
        y_context::AbstractArray,
        x_target::AbstractArray
    )

# Arguments
- `x_context::AbstractArray`: Locations of observed values of shape `(n, d, batch)`.
- `y_context::AbstractArray`: Observed values of shape `(n, channels, batch)`.
- `x_target::AbstractArray`: Locations of target set of shape `(m, d, batch)`.

# Returns
- `Tuple{AbstractArray, AbstractArray}`: Tuple containing means and variances.
"""
function (model::ConvCNP)(
    x_context::AbstractArray,
    y_context::AbstractArray,
    x_target::AbstractArray
)
    n_context = size(x_context, 1)

    # Compute discretisation of the functional embedding.
    x_discretisation = gpu(model.discretisation(x_context, x_target))

    if n_context > 0
        # The context set is non-empty. Compute encoding as usual.
        encoding = model.encoder(x_context, y_context, x_discretisation)
    else
        # The context set is empty. Set the encoding to all zeros.
        encoding = gpu(zeros(
            eltype(y_context),
            size(x_discretisation, 1),
            size(y_context, 2) + model.encoder.density,  # Account for density channel.
            size(y_context, 3)
        ))
    end

    # Apply the CNN. It operates on images of height one, so we have to insert a
    # dimension and pull it out afterwards.
    encoding = insert_dim(encoding; pos=2)
    latent = model.conv(encoding)
    if size(encoding, 1) != size(latent, 1) || size(latent, 2) != 1
        error(
            "Conv net changed the discretisation size from " *
            "$(size(encoding, 1)) to $(size(latent, 1))."
        )
    end
    latent = dropdims(latent; dims=2)

    # Perform decoding.
    channels = model.decoder(x_discretisation, latent, x_target)

    # Return predictive distribution.
    return model.predict(channels)
end

function _predict_gaussian_factorised(channels)
    # Check that the number of channels is even.
    mod(size(channels, 2), 2) != 0 && error("Number of channels must be even.")

    # Half of the channels are used to determine the mean, and the other half are used to
    # determine the standard deviation.
    i_split = div(size(channels, 2), 2)
    μ = channels[:, 1:i_split, :]
    σ² = NNlib.softplus.(channels[:, i_split + 1:end, :])
    return μ, σ²
end

"""
    convcnp_1d_factorised(arch::Architecture, margin::Float32=0.1f0)

Construct a ConvCNP for one-dimensional data with a factorised predictive distribution.

# Arguments
- `arch::Architecture`: CNN bundled with the points per units as constructed by
    `build_conv`.

# Keywords
- `margin::Float32=0.1f0`: Margin for the discretisation. See `UniformDiscretisation1d`.
"""
function convcnp_1d_factorised(arch::Architecture; margin::Float32=0.1f0)
    scale = 2 / arch.points_per_unit
    return ConvCNP(
        UniformDiscretisation1d(arch.points_per_unit, margin, arch.multiple),
        set_conv(1, scale; density=true),
        arch.conv,
        set_conv(2, scale; density=false),
        _predict_gaussian_factorised
    )
end

struct _PredictGaussianLowRank
    log_noise
end

@Flux.treelike _PredictGaussianLowRank

function (p::_PredictGaussianLowRank)(channels)
    # Get number of data points, channels, and batches.
    n, c, b = size(channels)

    μ = channels[:, 1:1, :]

    # Initialise the covariance with the observation noise.
    Σ = repeat(exp(p.log_noise) .* gpu(Matrix(I, n, n)), 1, 1, b)

    # Unfortunately, batched matrix multiplication does not have a gradient, so we do it
    # manually.
    for i = 2:c
        L = channels[:, i, :]
        Σ = Σ .+ reshape(L, n, 1, b) .* reshape(L, 1, n, b)
    end

    # Divide by the number of components to keep the scale of the variance right.
    Σ = Σ ./ (c - 1)  # Subtract one, because that is the mean.

    return μ, Σ
end

"""
    convcnp_1d_lowrank(arch::Architecture, margin::Float32=0.1f0)

Construct a ConvCNP for one-dimensional data with a low-rank predictive distribution.

# Arguments
- `arch::Architecture`: CNN bundled with the points per units as constructed by
    `build_conv`.

# Keywords
- `margin::Float32=0.1f0`: Margin for the discretisation. See `UniformDiscretisation1d`.
- `rank::Integer=rank`: Rank of the predictive covariance.
"""
function convcnp_1d_lowrank(
    arch::Architecture;
    margin::Float32=0.1f0,
    rank::Integer=10,
    noise::Float32=0.01f0
)
    scale = 2 / arch.points_per_unit
    return ConvCNP(
        UniformDiscretisation1d(arch.points_per_unit, margin, arch.multiple),
        set_conv(1, scale; density=true),
        arch.conv,
        set_conv(2 + rank, scale; density=false),
        _PredictGaussianLowRank(param(log(noise)))
    )
end
