export NP, np_1d, loglik, elbo, predict

"""
    abstract type AbstractNP

Abstract Neural Process type.
"""
abstract type AbstractNP end

"""
    struct NP <: AbstractNP

Neural Process.

# Fields
- `dim_embedding::Integer`: Dimensionality of the embedding.
- `encoder_det`: Deterministic encoder.
- `encoder_lat`: Latent encoder.
- `decoder`: Decoder.
- `log_σ²`: Natural logarithm of observation noise variance.
"""
struct NP <: AbstractNP
    dim_embedding::Integer
    encoder_det
    encoder_lat
    decoder
    log_σ²
end

@Flux.treelike NP

"""
    encoding_locations(model::NP, xc, xt)

Compute the locations for the latent encoding.

# Arguments
- `model::NP` Model.
- `xc`: Locations of context set of shape `(n, dims, batch)`.
- `xt`: Locations of target set of shape `(m, dims, batch)`.

# Returns
- `AbstractArray`: Locations of the encoding of shape `(k, dims, batch)`.
"""
encoding_locations(model::NP, xc, xt) = xt

"""
    encode_det(model::NP, xc, yc, xz)

Perform determistic encoding.

# Arguments
- `model::NP` Model.
- `xc`: Locations of context set of shape `(n, dims, batch)`.
- `yc`: Observed values of context set of shape `(n, channels, batch)`.
- `xz`: Locations of latent encoding of shape `(k, dims, batch)`.

# Returns
- `AbstractArray`: Deterministic encoding.
"""
encode_det(model::NP, xc, yc, xz) = model.encoder_det(xc, yc, xz)

"""
    empty_det_encoding(model::NP, xz)

Construct a deterministic encoding for the empty set.

# Arguments
- `model::NP` Model.
- `xz`: Locations of latent encoding of shape `(k, dims, batch)`.

# Returns
- `AbstractArray`: Empty deterministic encoding.
"""
function empty_det_encoding(model::NP, xz)
    k, _, batch_size = size(xz)
    return zeros(Float32, k, model.dim_embedding, batch_size) |> gpu
end

"""
    encode_lat(model::NP, xc, yc, xz)

Perform latent encoding.

# Arguments
- `model::NP` Model.
- `xc`: Locations of context set of shape `(n, dims, batch)`.
- `yc`: Observed values of context set of shape `(n, channels, batch)`.
- `xz`: Locations of latent encoding of shape `(k, dims, batch)`.

# Returns
- `Tuple{AbstractArray, AbstractArray}`: Tuple containing means and variances of shapes
    `(k, latent_channels, batch)`
"""
encode_lat(model::NP, xc, yc, xz) = split_μ_σ²(model.encoder_lat(xc, yc, xz))

"""
    empty_lat_encoding(model::NP, xz)

Construct a latent encoding for the empty set.

# Arguments
- `model::NP` Model.
- `xz`: Locations of latent encoding of shape `(k, dims, batch)`.

# Returns
- `AbstractArray`: Empty latent encoding.
"""
function empty_lat_encoding(model::NP, xz)
    k, _, batch_size = size(xz)
    μ = zeros(Float32, k, model.dim_embedding, batch_size) |> gpu
    σ² = ones(Float32, k, model.dim_embedding, batch_size) |> gpu
    return μ, σ²
end

"""
    decode(model::NP, xz, z, r, xt)

Perform decoding.

# Arguments
- `xz`: Locations of latent encoding of shape `(k, dims, batch)`.
- `z`: Samples of shape `(k, latent_channels, batch, num_samples)`.
- `r`: Deterministic encoding of shape `(k, dim_embedding, batch)`
- `xt`: Locations of target set of shape `(m, dims, batch)`.

# Returns
- `Tuple{AbstractArray, AbstractArray}`: Tuple containing means and variances.
"""
function decode(model::NP, xz, z, r, xt)
    num_samples = size(z, 4)
    # Repeat to be able to concatenate.
    return model.decoder(cat(
        repeat_gpu(r, 1, 1, 1, num_samples),
        z,
        repeat_gpu(xt, 1, 1, 1, num_samples),
        dims=2
    ))
end

"""
    struct NPEncoder

Encoder for a NP.

# Fields
- `mlp`: MLP.
"""
struct NPEncoder
    mlp
end

@Flux.treelike NPEncoder

"""
    (model::NPEncoder)(xc, yc, xt)

# Arguments
- `xc`: Locations of context set of shape `(n, dims, batch)`.
- `yc`: Observed values of context set of shape `(n, channels, batch)`.
- `xt`: Locations of target set of shape `(m, dims, batch)`.

# Returns
- `AbstractArray`: Encoding.
"""
function (model::NPEncoder)(xc, yc, xt)
    n_target = size(xt, 1)
    r = model.mlp(cat(xc, yc, dims=2))
    # Perform pooling operation and return tiled representation.
    return repeat_gpu(mean(r, dims=1), n_target, 1, 1)
end

"""
    (model::AbstractNP)(xc, yc, xt, num_samples::Integer)

# Arguments
- `xc`: Locations of context set of shape `(n, dims, batch)`.
- `yc`: Observed values of context set of shape `(n, channels, batch)`.
- `xt`: Locations of target set of shape `(m, dims, batch)`.
- `num_samples::Integer`: Number of samples.

# Returns
- `Tuple{AbstractArray, AbstractArray}`: Tuple containing means and variances.
"""

function (model::AbstractNP)(xc, yc, xt, num_samples::Integer)
    # Compute locations of the latent variable.
    xz = encoding_locations(model, xc, xt)

    # Compute deterministic encoding.
    if size(xc, 1) > 0
        r = encode_det(model, xc, yc, xz)
    else
        r = empty_det_encoding(model, xz)
    end

    # Construct prior over latent variable.
    if size(xc, 1) > 0
        pz = encode_lat(model, xc, yc, xz)
    else
        pz = empty_lat_encoding(model, xz)
    end

    # Sample latent variable.
    z = _sample(pz..., num_samples)

    # Perform decoding.
    channels = decode(model, xz, z, r, xt)

    # Return the predictions with noise.
    return channels, exp.(model.log_σ²)
end

function _sample(μ, σ², num_samples)
    noise = randn(Float32, size(μ)..., num_samples) |> gpu
    return μ .+ sqrt.(σ²) .* noise
end

"""
    np_1d(;
        dim_embedding::Integer,
        num_encoder_layers::Integer,
        num_decoder_layers::Integer,
        σ²::Float32=1f-4
    )

# Arguments
- `dim_embedding::Integer`: Dimensionality of the embedding.
- `num_encoder_layers::Integer`: Number of layers in the encoder.
- `num_decoder_layers::Integer`: Number of layers in the decoder.
- `σ²::Float32=1f-4`: Initialisation of the observation noise variance.

# Returns
- `NP`: Corresponding model.
"""
function np_1d(;
    dim_embedding::Integer,
    num_encoder_layers::Integer,
    num_decoder_layers::Integer,
    σ²::Float32=1f-4
)
    dim_x = 1
    dim_y = 1
    return NP(
        dim_embedding,
        NPEncoder(
            batched_mlp(
                dim_x + dim_y,
                dim_embedding,
                dim_embedding,
                num_encoder_layers
            )
        ),
        NPEncoder(
            batched_mlp(
                dim_x + dim_y,
                2dim_embedding,
                2dim_embedding,
                num_encoder_layers
            )
        ),
        batched_mlp(
            2dim_embedding + dim_x,
            dim_embedding,
            dim_y,
            num_decoder_layers
        ),
        param([log(σ²)])
    )
end

"""
    loglik(model::AbstractNP, epoch::Integer, xc, yc, xt, yt; num_samples::Integer)

Log-expected-likelihood loss. This is a biased estimate of the log-likelihood.

# Arguments
- `model::AbstractNP`: Model.
- `epoch::Integer`: Current epoch.
- `xc`: Locations of context set of shape `(n, dims, batch)`.
- `yc`: Observed values of context set of shape `(n, channels, batch)`.
- `xt`: Locations of target set of shape `(m, dims, batch)`.
- `yt`: Observed values of target set of shape `(m, channels, batch)`.

# Keywords
- `num_samples::Integer`: Number of samples.

# Returns
- `Real`: Average negative log-expected likelihood.
"""
function loglik(model::AbstractNP, epoch::Integer, xc, yc, xt, yt; num_samples::Integer)
    μ, σ² = model(xc, yc, xt, num_samples)

    # Compute the components of the ELBO.
    logpdfs = gaussian_logpdf(yt, μ, σ²)

    # Sum over data points and channels to assemble the log-pdfs.
    logpdfs = sum(logpdfs, dims=(1, 2))

    # Log-mean-exp over samples.
    logpdfs = logsumexp(logpdfs, dims=4) .- Float32(log(num_samples))

    # Return average over batches.
    return -mean(logpdfs)
end

"""
    elbo(model::AbstractNP, epoch::Integer, xc, yc, xt, yt, num_samples::Integer)

Neural process ELBO-style loss.

# Arguments
- `model::AbstractNP`: Model.
- `epoch::Integer`: Current epoch.
- `xc`: Locations of context set of shape `(n, dims, batch)`.
- `yc`: Observed values of context set of shape `(n, channels, batch)`.
- `xt`: Locations of target set of shape `(m, dims, batch)`.
- `yt`: Observed values of target set of shape `(m, channels, batch)`.

# Keywords
- `num_samples::Integer`: Number of samples.

# Returns
- `Real`: Average negative NP loss.
"""
function elbo(model::AbstractNP, epoch::Integer, xc, yc, xt, yt; num_samples::Integer)
    xz = encoding_locations(model, xc, xt)

    # Compute deterministic encoding.
    if size(xc, 1) > 0
        r = encode_det(model, xc, yc, xz)
    else
        r = empty_det_encoding(model, xz)
    end

    # Construct prior over latent variable.
    if size(xc, 1) > 0
        pz = encode_lat(model, xc, yc, xz)
    else
        pz = empty_lat_encoding(model, xz)
    end

    # Construct posterior over latent variable.
    qz = encode_lat(
        model,
        cat(xc, xt, dims=1),
        cat(yc, yt, dims=1),
        xz
    )

    # Sample latent variable and compute predictive statistics.
    samples = _sample(qz..., num_samples)
    μ = decode(model, xz, samples, r, xt)
    σ² = exp.(model.log_σ²)

    # Compute the components of the ELBO.
    expectations = gaussian_logpdf(yt, μ, σ²)
    kls = kl(qz..., pz...)

    # Sum over data points and channels to assemble the expressions.
    expectations = sum(expectations, dims=(1, 2))
    kls = sum(kls, dims=(1, 2))

    # Estimate ELBO from samples.
    elbos = mean(expectations, dims=4) .- kls

    # Return average over batches.
    return -mean(elbos)
end

"""
    predict(
        model::AbstractNP,
        xc::AbstractVector,
        yc::AbstractVector,
        xt::AbstractVector;
        num_samples::Integer=10
    )

# Arguments
- `model::AbstractNP`: Model.
- `xc::AbstractVector`: Locations of observed values of shape `(n)`.
- `yc::AbstractVector`: Observed values of shape `(n)`.
- `xt::AbstractVector`: Locations of target values of shape `(m)`.

# Keywords
- `num_samples::Integer=10`: Number of posterior samples.

# Returns
- `Tuple{Nothing, Nothing, Nothing, AbstractArray}`: Tuple containing `nothing`, `nothing`,
    `nothing`, and `num_samples` posterior samples.
"""
function predict(
    model::AbstractNP,
    xc::AbstractVector,
    yc::AbstractVector,
    xt::AbstractVector;
    num_samples::Integer=10
)
    μ, σ² = untrack(model)(expand_gpu.((xc, yc, xt))..., num_samples)
    samples = μ[:, 1, 1, :] |> cpu
    return nothing, nothing, nothing, samples
end
