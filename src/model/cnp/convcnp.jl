export ConvCNP, convcnp_1d, loglik, predict

"""
    ConvCNP

Convolutional CNP model with a factorised Gaussian predictive distribution.

# Fields
- `disc::Discretisation`: Discretisation for the encoding.
- `encoder::SetConv`: Encoder.
- `conv::Chain`: CNN that approximates ρ.
- `decoder::SetConv`: Decoder.
- `predict`: Function that transforms the decoding into a predictive distribution.
"""
struct ConvCNP
    disc::Discretisation
    encoder::SetConv
    conv::Chain
    decoder::SetConv
    predict
end

@Flux.treelike ConvCNP

"""
    (model::ConvCNP)(xc, yc, xt)

# Arguments
- `xc`: Locations of observed values of shape `(n, d, batch)`.
- `yc`: Observed values of shape `(n, channels, batch)`.
- `xt`: Locations of target set of shape `(m, d, batch)`.

# Returns
- `Tuple{AbstractArray, AbstractArray}`: Tuple containing means and variances.
"""
function (model::ConvCNP)(xc, yc, xt)
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
        split_μ_σ²
    )
end

"""
    loglik(model::ConvCNP, epoch::Integer, xc, yc, xt, yt)

# Arguments
- `model::ConvCNP`: Model.
- `epoch::Integer`: Current epoch.
- `xc`: Locations of observed values of shape `(n, d, batch)`.
- `yc`: Observed values of shape `(n, channels, batch)`.
- `xt`: Locations of target values of shape `(m, d, batch)`.
- `yt`: Target values of shape `(m, channels, batch)`.

# Returns
- `Real`: Average negative log-likelihood.
"""
function loglik(model::ConvCNP, epoch::Integer, xc, yc, xt, yt)
    logpdfs = gaussian_logpdf(yt, model(xc, yc, xt)...)
    # Sum over data points before averaging over tasks.
    return -mean(sum(logpdfs, dims=1))
end

"""
    predict(
        model::ConvCNP,
        xc::AbstractVector,
        yc::AbstractVector,
        xt::AbstractVector
    )

# Arguments
- `model::ConvCNP`: Model.
- `xc`: Locations of observed values of shape `(n, d, batch)`.
- `yc`: Observed values of shape `(n, channels, batch)`.
- `xt`: Locations of target values of shape `(m, d, batch)`.

# Returns
- `Tuple{AbstractArray, AbstractArray, AbstractArray, Nothing}`: Tuple containing means,
    lower and upper 95% central credible bounds, and `nothing`.
"""
function predict(
    model::ConvCNP,
    xc::AbstractVector,
    yc::AbstractVector,
    xt::AbstractVector
)
    μ, σ² = untrack(model)(expand_gpu.((xc, yc, xt))...)
    μ = μ[:, 1, 1] |> cpu
    σ² = σ²[:, 1, 1] |> cpu
    return μ, μ .- 2 .* sqrt.(σ²), μ .+ 2 .* sqrt.(σ²), nothing
end
