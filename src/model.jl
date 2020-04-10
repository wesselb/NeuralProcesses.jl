export ConvCNP, convcnp_1d_factorised, convcnp_1d_lowrank, convcnp_1d_kernel

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

function _predict_gaussian_lowrank(channels)
    # Get number of data points, channels, and batches.
    n, c, b = size(channels)

    μ = channels[:, 1:1, :]

    # Initialise the covariance with heterogeneous observation noise.
    noise_channel = NNlib.softplus.(channels[:, 2, :])
    Σ = cat([diagonal(noise_channel[:, i]) for i = 1:b]..., dims=3)

    # Unfortunately, batched matrix multiplication does not have a gradient, so we do it
    # manually.
    for i = 3:c
        L = channels[:, i, :]
        Σ = Σ .+ reshape(L, n, 1, b) .* reshape(L, 1, n, b)
    end

    # Divide by the number of components to keep the scale of the variance right.
    Σ = Σ ./ (c - 1)  # Subtract one to account for the mean.

    return μ, Σ
end

"""
    convcnp_1d_lowrank(
        arch::Architecture;
        margin::Float32=0.1f0,
        rank::Integer=10
    )

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
    rank::Integer=10
)
    scale = 2 / arch.points_per_unit
    return ConvCNP(
        UniformDiscretisation1d(arch.points_per_unit, margin, arch.multiple),
        set_conv(1, scale; density=true),
        arch.conv,
        set_conv(2 + rank, scale; density=false),
        _predict_gaussian_lowrank
    )
end


struct ConvCNPKernel
    mean_discretisation::Discretisation
    kernel_discretisation::Discretisation
    mean_encoder::SetConv
    kernel_encoder::SetConv
    mean_conv::Chain
    kernel_conv::Chain
    mean_decoder::SetConv
    kernel_decoder::SetConv
    predict
end

@Flux.treelike ConvCNPKernel

function (model::ConvCNPKernel)(
    x_context::AbstractArray,
    y_context::AbstractArray,
    x_target::AbstractArray
)
    n_context = size(x_context, 1)

    # Compute discretisation of the functional embedding.
    mean_discretisation = gpu(model.mean_discretisation(x_context, x_target))
    kernel_discretisation = gpu(model.kernel_discretisation(x_context, x_target))

    if n_context > 0
        # The context set is non-empty. Compute encodings as usual.
        mean_encoding = model.mean_encoder(x_context, y_context, mean_discretisation)
        kernel_encoding = kernel(model.kernel_encoder, x_context, y_context, kernel_discretisation)
    else
        # The context set is empty. Set the encodings to all zeros.
        mean_encoding = gpu(zeros(
            eltype(y_context),
            size(mean_discretisation, 1),
            size(y_context, 2) + model.encoder.density,  # Account for density channel.
            size(y_context, 3)
        ))
        kernel_encoding = gpu(zeros(
            eltype(y_context),
            size(kernel_discretisation, 1),
            size(kernel_discretisation, 1),
            size(y_context, 2) + model.encoder.density, # Account for density channel.
            size(y_context, 3)
        ))
        # Append identity channel.
        identity = gpu(repeat(Matrix{Float32}(
            I,
            size(kernel_discretisation, 1),
            size(kernel_discretisation, 1)
        ), 1, 1, 1, size(y_context, 3)))
        kernel_encoding = cat(kernel_encoding, identity, dims=3)

    end

    # Apply the mean CNN. It operates on images of height one, so we have to insert a
    # dimension and pull it out afterwards.
    mean_encoding = insert_dim(mean_encoding; pos=2)
    mean_latent = model.mean_conv(mean_encoding)
    if size(mean_encoding, 1) != size(mean_latent, 1) || size(mean_latent, 2) != 1
        error(
            "Mean conv net changed the discretisation size from " *
            "$(size(mean_encoding, 1)) to $(size(mean_latent, 1))."
        )
    end
    mean_latent = dropdims(mean_latent; dims=2)

    # Apply the kernel CNN.
    kernel_latent = model.kernel_conv(kernel_encoding)

    # Perform decoding.
    mean_channels = model.mean_decoder(mean_discretisation, mean_latent, x_target)
    kernel_channels = kernel_smooth(model.kernel_decoder, kernel_discretisation, kernel_latent, x_target)

    # Return predictive distribution.
    return model.predict(mean_channels, kernel_channels)
end

function convcnp_1d_kernel(
    mean_arch::Architecture,
    kernel_arch::Architecture;
    margin::Float32=0.1f0
)
    mean_scale = 2 / mean_arch.points_per_unit
    kernel_scale = 2 / kernel_arch.points_per_unit
    return ConvCNPKernel(
        UniformDiscretisation1d(mean_arch.points_per_unit, margin, mean_arch.multiple),
        UniformDiscretisation1d(kernel_arch.points_per_unit, margin, kernel_arch.multiple),
        set_conv(1, mean_scale; density=true),
        set_conv(1, kernel_scale; density=true),
        mean_arch.conv,
        kernel_arch.conv,
        set_conv(1, mean_scale; density=false),
        set_conv(1, kernel_scale; density=false),
        _predict_gaussian_kernel
    )
end

function _predict_gaussian_kernel(mean_channels, kernel_channels)
    length(size(mean_channels)) == 3 || error("Mean tensor must be rank 3.")
    length(size(kernel_channels)) == 4 || error("Kernel tensor must be rank 4.")
    size(mean_channels, 2) == 1 || error("Mean tensor must have exactly one channel.")
    size(kernel_channels, 3) == 1 || error("Kernel tensor must have exactly one channel.")

    return mean_channels[:, 1, :], kernel_channels[:, :, 1, :]
end
