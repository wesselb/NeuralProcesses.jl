export convcnp_1d_correlated

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
    (model::CorrelatedConvCNP)(model::CorrelatedConvCNP)(xc::AA, yc::AA, xt::AA)

# Arguments
- `xc::AA`: Locations of observed values of shape `(n, d, batch)`.
- `yc::AA`: Observed values of shape `(n, channels, batch)`.
- `xt::AA`: Locations of target set of shape `(m, d, batch)`.

# Returns
- `Tuple{AA, AA}`: Tuple containing means and covariances.
"""
function (model::CorrelatedConvCNP)(xc::AA, yc::AA, xt::AA)
    if !isnothing(xc) && size(xc, 1) > 0
        # The context set is non-empty.
        μ_xz = model.μ_disc(xc, xt) |> gpu
        Σ_xz = model.Σ_disc(xc, xt) |> gpu
        μ_channels = encode(   model.μ_encoder, xc, yc, μ_xz)
        Σ_channels = encode_pd(model.Σ_encoder, xc, yc, Σ_xz)
    else
        # The context set is empty.
        μ_xz = model.μ_disc(xt) |> gpu
        Σ_xz = model.Σ_disc(xt) |> gpu
        μ_channels = empty_encoding(   model.μ_encoder, μ_xz)
        Σ_channels = empty_encoding_pd(model.Σ_encoder, Σ_xz)
    end

    # Apply the CNNs.
    μ_channels = with_dummy(model.μ_conv, μ_channels)
    Σ_channels = model.Σ_conv(Σ_channels)

    # Perform decoding.
    μ_channels = decode(   model.μ_decoder, μ_xz, μ_channels, xt)
    Σ_channels = decode_pd(model.Σ_decoder, Σ_xz, Σ_channels, xt)

    # Return predictive distribution.
    return model.predict(μ_channels, Σ_channels)
end

"""
    convcnp_1d_correlated(
        μ_arch::Architecture,
        Σ_arch::Architecture;
        margin::Float32=0.1f0
    )

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
    loglik(model::CorrelatedConvCNP, epoch::Integer, xc::AA, yc::AA, xt::AA, yt::AA)

# Arguments
- `model::CorrelatedConvCNP`: Model.
- `epoch::Integer`: Current epoch.
- `xc::AA`: Locations of observed values of shape `(n, d, batch)`.
- `yc::AA`: Observed values of shape `(n, channels, batch)`.
- `xt::AA`: Locations of target values of shape `(m, d, batch)`.
- `yt::AA`: Target values of shape `(m, channels, batch)`.

# Returns
- `Real`: Average negative log-likelihood.
"""
function loglik(model::CorrelatedConvCNP, epoch::Integer, xc::AA, yc::AA, xt::AA, yt::AA)
    size(yt, 2) == 1 || error("Target outputs have more than one channel.")

    n_target, _, batch_size = size(xt)

    μ, Σ = model(xc, yc, xt)

    logpdf = 0f0
    ridge = gpu(Matrix(_epoch_to_reg(epoch) * I, n_target, n_target))
    for i = 1:batch_size
        logpdf += gaussian_logpdf(yt[:, 1, i], μ[:, i], Σ[:, :, i] .+ ridge)
    end

    return -logpdf / batch_size
end

_epoch_to_reg(epoch) = 10^(-min(1 + Float32(epoch), 5))

"""
    predict(model::CorrelatedConvCNP, xc::AV, yc::AV, xt::AV)

# Arguments
- `model::CorrelatedConvCNP`: Model.
- `xc`: Locations of observed values of shape `(n, d, batch)`.
- `yc`: Observed values of shape `(n, channels, batch)`.
- `xt`: Locations of target values of shape `(m, d, batch)`.

# Returns
- `Tuple{AA, AA, AA, AA}`: Tuple containing means, lower and upper 95% central credible
    bounds, and three posterior samples.
"""
function predict(model::CorrelatedConvCNP, xc::AV, yc::AV, xt::AV)
    μ, Σ = untrack(model)(expand_gpu.((xc, yc, xt)))
    μ = μ[:, 1, 1] |> cpu
    Σ = Σ[:, :, 1] |> cpu
    σ = sqrt.(diag(Σ))

    # Produce three posterior samples.
    samples = cholesky(y_cov).U' * randn(length(x), 3) .+ y_mean

    return μ, μ .- 2 .* σ, μ .+ 2 .* σ, samples
end
