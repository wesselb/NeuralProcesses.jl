export ConvCNP, convcnp_1d, loglik, predict

"""
    ConvCNP

Convolutional CNP model with a factorised Gaussian predictive distribution.

# Fields
- `disc::Discretisation`: Discretisation for the encoding.
- `encoder`: Encoder.
- `conv::Chain`: CNN that approximates ρ.
- `decoder`: Decoder.
- `predict`: Function that transforms the decoding into a predictive distribution.
"""
struct ConvCNP
    disc::Discretisation
    encoder
    conv::Chain
    decoder
    predict
end

@Flux.treelike ConvCNP

"""
    (model::ConvCNP)(xc::MaybeAA, yc::MaybeAA, xt::AA)

# Arguments
- `xc::MaybeAA`: Locations of observed values of shape `(n, d, batch)`.
- `yc::MaybeAA`: Observed values of shape `(n, channels, batch)`.
- `xt::AA`: Locations of target set of shape `(m, d, batch)`.

# Returns
- `Tuple{AA, AA}`: Tuple containing means and standard deviations.
"""
function (model::ConvCNP)(xc::MaybeAA, yc::MaybeAA, xt::AA)
    if !isnothing(xc) && size(xc, 1) > 0
        # The context set is non-empty.
        xz = model.disc(xc, xt) |> gpu
        channels = encode(model.encoder, xc, yc, xz)
    else
        # The context set is empty.
        xz = model.disc(xt) |> gpu
        channels = empty_encoding(model.encoder, xz)
    end

    # Apply CNN.
    channels = with_dummy(model.conv, channels)

    # Perform decoding.
    channels = decode(model.decoder, xz, channels, xt)

    # Return predictive distribution.
    return model.predict(channels)
end

"""
    convcnp_1d(;
        receptive_field::Float32,
        num_layers::Integer,
        num_channels::Integer,
        points_per_unit::Float32,
        margin::Float32=receptive_field
    )

Construct a ConvCNP for one-dimensional data.

# Keywords
- `receptive_field::Float32`: Width of the receptive field.
- `num_layers::Integer`: Number of layers of the CNN, excluding an initial
    and final pointwise convolutional layer to change the number of channels
    appropriately.
- `num_channels::Integer`: Number of channels of the CNN.
- `margin::Float32=receptive_field`: Margin for the discretisation. See
    `UniformDiscretisation1d`.
"""
function convcnp_1d(;
    receptive_field::Float32,
    num_layers::Integer,
    num_channels::Integer,
    points_per_unit::Float32,
    margin::Float32=receptive_field
)
    # Build architecture.
    arch = build_conv(
        receptive_field,
        num_layers,
        num_channels,
        points_per_unit=points_per_unit,
        dimensionality=1,
        num_in_channels=2,  # Account for density channel.
        num_out_channels=2
    )

    scale = 2 / arch.points_per_unit
    return ConvCNP(
        UniformDiscretisation1d(arch.points_per_unit, margin, arch.multiple),
        set_conv(2, scale),  # Account for density channel.
        arch.conv,
        set_conv(2, scale),
        split_μ_σ
    )
end

"""
    loglik(model::ConvCNP, epoch::Integer, xc::AA, yc::AA, xt::AA, yt::AA)

# Arguments
- `model::ConvCNP`: Model.
- `epoch::Integer`: Current epoch.
- `xc::AA`: Locations of observed values of shape `(n, d, batch)`.
- `yc::AA`: Observed values of shape `(n, channels, batch)`.
- `xt::AA`: Locations of target values of shape `(m, d, batch)`.
- `yt::AA`: Target values of shape `(m, channels, batch)`.

# Returns
- `Real`: Average negative log-likelihood.
"""
function loglik(model::ConvCNP, epoch::Integer, xc::AA, yc::AA, xt::AA, yt::AA)
    logpdfs = gaussian_logpdf(yt, model(xc, yc, xt)...)
    # Sum over data points before averaging over tasks.
    return -mean(sum(logpdfs, dims=1))
end

"""
    predict(model::ConvCNP, xc::AV, yc::AV, xt::AV)

# Arguments
- `model::ConvCNP`: Model.
- `xc::AV`: Locations of observed values of shape `(n, d, batch)`.
- `yc::AV`: Observed values of shape `(n, channels, batch)`.
- `xt::AV`: Locations of target values of shape `(m, d, batch)`.

# Returns
- `Tuple{AA, AA, AA, Nothing}`: Tuple containing means, lower and upper 95% central
    credible bounds, and `nothing`.
"""
function predict(model::ConvCNP, xc::AV, yc::AV, xt::AV)
    μ, σ = untrack(model)(expand_gpu.((xc, yc, xt))...)
    μ = μ[:, 1, 1] |> cpu
    σ = σ[:, 1, 1] |> cpu
    return μ, μ .- 2 .* σ, μ .+ 2 .* σ, nothing
end
