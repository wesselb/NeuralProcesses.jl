export CorrelatedConvCNP, convcnp_1d_correlated, loss, predict

"""
    CorrelatedConvCNP

Convolutional CNP model with a correlated Gaussian predictive distribution.

# Fields
- `μ_disc::Discretisation`: Discretisation for the encoding of the mean.
- `Σ_disc::Discretisation`: Discretisation for the encoding of the covariance.
- `μ_encoder::SetConv`: Encoder for the mean.
- `Σ_encoder::SetConv`: Encoder for the covariance.
- `μ_conv::Chain`: CNN that approximates ρ for the mean.
- `Σ_conv::Chain`: CNN that approximates ρ for the covariance.
- `μ_decoder::SetConv`: Decoder for the mean.
- `Σ_decoder::SetConv`: Decoder for the covariance.
- `predict`: Function that transforms the decodings into a predictive distribution.
"""
struct CorrelatedConvCNP
    μ_disc::Discretisation
    Σ_disc::Discretisation
    μ_encoder::SetConv
    Σ_encoder::SetConv
    μ_conv::Chain
    Σ_conv::Chain
    μ_decoder::SetConv
    Σ_decoder::SetConv
    predict
end

@Flux.treelike CorrelatedConvCNP

"""
    (model::CorrelatedConvCNP)(
        x_context::AbstractArray,
        y_context::AbstractArray,
        x_target::AbstractArray
    )

# Arguments
- `x_context::AbstractArray`: Locations of observed values of shape `(n, d, batch)`.
- `y_context::AbstractArray`: Observed values of shape `(n, channels, batch)`.
- `x_target::AbstractArray`: Locations of target set of shape `(m, d, batch)`.

# Returns
- `Tuple{AbstractArray, AbstractArray}`: Tuple containing means and covariances.
"""
function (model::CorrelatedConvCNP)(
    x_context::AbstractArray,
    y_context::AbstractArray,
    x_target::AbstractArray
)
    # Compute discretisation of the functional embedding.
    μ_x_latent = model.μ_disc(x_context, x_target) |> gpu
    Σ_x_latent = model.Σ_disc(x_context, x_target) |> gpu

    if size(x_context, 1) > 0
        # The context set is non-empty. Compute encodings as usual.
        μ_channels = encode(   model.μ_encoder, x_context, y_context, μ_x_latent)
        Σ_channels = encode_pd(model.Σ_encoder, x_context, y_context, Σ_x_latent)
    else
        # The context set is empty. Set to empty encodings.
        μ_channels = empty_encoding(   model.μ_encoder, μ_x_latent)
        Σ_channels = empty_encoding_pd(model.Σ_encoder, Σ_x_latent)
    end

    # Apply the CNNs.
    μ_channels = model.μ_conv(μ_channels)
    Σ_channels = model.Σ_conv(Σ_channels)

    # Perform decoding.
    μ_channels = decode(   model.μ_decoder, μ_x_latent, μ_channels, x_target)
    Σ_channels = decode_pd(model.Σ_decoder, Σ_x_latent, Σ_channels, x_target)

    # Return predictive distribution.
    return model.predict(μ_channels, Σ_channels)
end

"""
    convcnp_1d_correlated(arch::Architecture, margin::Float32=0.1f0)

Construct a correlated ConvCNP for one-dimensional data.

# Arguments
- `arch::Architecture`: CNN bundled with the points per units as constructed by
    `build_conv`.

# Keywords
- `margin::Float32=0.1f0`: Margin for the discretisation. See `UniformDiscretisation1d`.
"""
function convcnp_1d_correlated(
    μ_arch::Architecture,
    Σ_arch::Architecture;
    margin::Float32=0.1f0
)
    μ_scale = 2 / μ_arch.points_per_unit
    Σ_scale = 2 / Σ_arch.points_per_unit
    return CorrelatedConvCNP(
        UniformDiscretisation1d(μ_arch.points_per_unit, margin, μ_arch.multiple),
        UniformDiscretisation1d(Σ_arch.points_per_unit, margin, Σ_arch.multiple),
        set_conv(2, μ_scale),  # Account for density channel.
        set_conv(2, Σ_scale),  # Account for density channel.
        μ_arch.conv,
        Σ_arch.conv,
        set_conv(1, μ_scale),
        set_conv(1, Σ_scale),
        _predict_gaussian_correlated
    )
end

function _predict_gaussian_correlated(μ_channels, Σ_channels)
    size(μ_channels, 2) == 1 || error("Mean is not one-dimensional.")
    size(μ_channels, 3) == 1 || error("More than one channel for the mean.")
    size(Σ_channels, 3) == 1 || error("More than one channel for the covariance.")
    return μ_channels[:, 1, 1, :], Σ_channels[:, :, 1, :]
end

"""
    loss(
        model::CorrelatedConvCNP,
        epoch::Integer,
        x_context::AbstractArray,
        y_context::AbstractArray,
        x_target::AbstractArray,
        y_target::AbstractArray
    )

# Arguments
- `model::CorrelatedConvCNP`: Model.
- `epoch::Integer`: Current epoch.
- `x_context::AbstractArray`: Locations of observed values of shape `(n, d, batch)`.
- `y_context::AbstractArray`: Observed values of shape `(n, channels, batch)`.
- `x_target::AbstractArray`: Locations of target values of shape `(m, d, batch)`.
- `y_target::AbstractArray`: Target values of shape `(m, channels, batch)`.

# Returns
- `Real`: Average negative log-likelihood.
"""
function loss(
    model::CorrelatedConvCNP,
    epoch::Integer,
    x_context::AbstractArray,
    y_context::AbstractArray,
    x_target::AbstractArray,
    y_target::AbstractArray
)
    size(y_target, 2) == 1 || error("Target outputs have more than one channel.")

    n_target, _, batch_size = size(x_target)

    μ, Σ = model(x_context, y_context, x_target)

    logpdf = 0f0
    ridge = gpu(Matrix(_epoch_to_reg(epoch) * I, n_target, n_target))
    for i = 1:batch_size
        logpdf += gaussian_logpdf(y_target[:, 1, i], μ[:, i], Σ[:, :, i] .+ ridge)
    end

    return -logpdf / batch_size
end

_epoch_to_reg(epoch) = 10^(-min(1 + Float32(epoch), 5))

"""
    predict(
        model::CorrelatedConvCNP,
        x_context::AbstractVector,
        y_context::AbstractVector,
        x_target::AbstractVector
    )

# Arguments
- `model::CorrelatedConvCNP`: Model.
- `x_context::AbstractArray`: Locations of observed values of shape `(n, d, batch)`.
- `y_context::AbstractArray`: Observed values of shape `(n, channels, batch)`.
- `x_target::AbstractArray`: Locations of target values of shape `(m, d, batch)`.

# Returns
- `Tuple{AbstractArray, AbstractArray, AbstractArray, AbstractArray}`: Tuple containing
    means, lower and upper 95% central credible bounds, and three posterior samples.
"""
function predict(
    model::CorrelatedConvCNP,
    x_context::AbstractVector,
    y_context::AbstractVector,
    x_target::AbstractVector
)
    μ, Σ = untrack(model)(_expand_gpu.((x_context, y_context, x_target)))
    μ = μ[:, 1, 1] |> cpu
    Σ = Σ[:, :, 1] |> cpu
    σ² = diag(Σ)

    # Produce three posterior samples.
    samples = cholesky(y_cov).U' * randn(length(x), 3) .+ y_mean

    return μ, μ .- 2 .* sqrt.(σ²), μ .+ 2 .* sqrt.(σ²), samples
end
