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
- `encoder_lat`: Latent encoder.
- `encoder_det`: Deterministic encoder.
- `decoder`: Decoder.
- `log_σ`: Natural logarithm of observation noise.
"""
struct NP <: AbstractNP
    encoder_lat
    encoder_det
    decoder
    log_σ
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
empty_det_encoding(model::NP, xz) = empty_encoding(model.encoder_det, xz)

"""
    encode_lat(model::NP, xc, yc, xz)

Perform latent encoding.

# Arguments
- `model::NP` Model.
- `xc`: Locations of context set of shape `(n, dims, batch)`.
- `yc`: Observed values of context set of shape `(n, channels, batch)`.
- `xz`: Locations of latent encoding of shape `(k, dims, batch)`.

# Returns
- `Tuple{AbstractArray, AbstractArray}`: Tuple containing means and standard deviations of
    shapes `(k, latent_channels, batch)`
"""
encode_lat(model::NP, xc, yc, xz) = split_μ_σ(model.encoder_lat(xc, yc, xz))

"""
    empty_lat_encoding(model::NP, xz)

Construct a latent encoding for the empty set.

# Arguments
- `model::NP` Model.
- `xz`: Locations of latent encoding of shape `(k, dims, batch)`.

# Returns
- `AbstractArray`: Empty latent encoding.
"""
empty_lat_encoding(model::NP, xz) = split_μ_σ(empty_encoding(model.encoder_lat, xz))

"""
    encode(model::AbstractNP, xc, yc, xz)

Perform determistic and latent encoding.

# Arguments
- `model::AbstractNP` Model.
- `xc`: Locations of context set of shape `(n, dims, batch)`.
- `yc`: Observed values of context set of shape `(n, channels, batch)`.
- `xz`: Locations of latent encoding of shape `(k, dims, batch)`.

# Returns
- `Tuple`: Tuple containing locations of encodings, the deterministic encoding,
    and the latent encoding.
"""
function encode(model::AbstractNP, xc, yc, xt)
    # Compute locations of the encodings.
    xz = encoding_locations(model, xc, xt)

    # Compute deterministic and latent encoding.
    if size(xc, 1) > 0
        # Context set is non-empty.
        return xz, encode_lat(model, xc, yc, xz), encode_det(model, xc, yc, xz)
    else
        # Context set is empty.
        return xz, empty_lat_encoding(model, xz), empty_det_encoding(model, xz)
    end
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
- `Tuple{AbstractArray, AbstractArray}`: Tuple containing means and standard deviations.
"""
decode(model::NP, xz, z, r, xt) = model.decoder(repeat_cat(z, r, xt, dims=2))

"""
    (model::AbstractNP)(xc, yc, xt, num_samples::Integer)

# Arguments
- `xc`: Locations of context set of shape `(n, dims, batch)`.
- `yc`: Observed values of context set of shape `(n, channels, batch)`.
- `xt`: Locations of target set of shape `(m, dims, batch)`.
- `num_samples::Integer`: Number of samples.

# Returns
- `Tuple{AbstractArray, AbstractArray}`: Tuple containing means and standard deviations.
"""

function (model::AbstractNP)(xc, yc, xt, num_samples::Integer)
    # Perform deterministic and latent encoding.
    xz, pz, r = encode(model, xc, yc, xt)

    # Sample latent variable.
    z = _sample(pz..., num_samples)

    # Perform decoding.
    channels = decode(model, xz, z, r, xt)

    # Return the predictions with noise.
    return channels, exp.(model.log_σ)
end

function _sample(μ::AbstractArray, σ::AbstractArray, num_samples::Integer)
    ε = randn_gpu(Float32, size(μ)..., num_samples)
    return μ .+ σ .* ε
end

_sample(d₁::Tuple, d₂::Tuple, num_samples::Integer) =
    (_sample(d₁..., num_samples), _sample(d₂..., num_samples))

"""
    struct NPEncoder

Encoder for a NP.

# Fields
- `ff₁`: Pre-pooling feed-forward net.
- `ff₂`: Post-pooling feed-forward net.
"""
struct NPEncoder
    ff₁
    ff₂
end

@Flux.treelike NPEncoder

"""
    (model::NPEncoder)(xc, yc, xz)

# Arguments
- `xc`: Locations of context set of shape `(n, dims, batch)`.
- `yc`: Observed values of context set of shape `(n, channels, batch)`.
- `xz`: Locations of latent encoding of shape `(k, dims, batch)`.

# Returns
- `AbstractArray`: Encoding.
"""
(encoder::NPEncoder)(xc, yc, xz) =
    encoder.ff₂(mean(encoder.ff₁(cat(xc, yc, dims=2)), dims=1))

"""
    empty_encoding(encoder::NPEncoder, xz)

Construct an encoding for the empty set.

# Arguments
- `encoder::NPEncoder` Model.
- `xz`: Locations of encoding of shape `(k, dims, batch)`.

# Returns
- `AbstractArray`: Empty encoding.
"""
function empty_encoding(encoder::NPEncoder, xz)
    batch_size = size(xz, 3)
    r = zeros_gpu(Float32, 1, encoder.ff₁.dim_out, batch_size)
    return encoder.ff₂(r)
end

"""
    np_1d(;
        dim_embedding::Integer,
        num_encoder_layers::Integer,
        num_decoder_layers::Integer,
        σ::Float32=1f-2,
        learn_σ::Bool=true
    )

# Arguments
- `dim_embedding::Integer`: Dimensionality of the embedding.
- `num_encoder_layers::Integer`: Number of layers in the encoder.
- `num_decoder_layers::Integer`: Number of layers in the decoder.
- `σ::Float32=1f-2`: Initialisation of the observation noise.
- `learn_σ::Bool=true`: Learn the observation noise.

# Returns
- `NP`: Corresponding model.
"""
function np_1d(;
    dim_embedding::Integer,
    num_encoder_layers::Integer,
    num_decoder_layers::Integer,
    σ::Float32=1f-2,
    learn_σ::Bool=true
)
    dim_x = 1
    dim_y = 1
    return NP(
        NPEncoder(
            batched_mlp(
                dim_in    =dim_x + dim_y,
                dim_hidden=dim_embedding,
                dim_out   =dim_embedding,
                num_layers=num_encoder_layers
            ),
            batched_mlp(
                dim_in    =dim_embedding,
                dim_hidden=dim_embedding,
                dim_out   =2dim_embedding,
                num_layers=2
            )
        ),
        NPEncoder(
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
                num_layers=2
            )
        ),
        batched_mlp(
            dim_in    =2dim_embedding + dim_x,
            dim_hidden=dim_embedding,
            dim_out   =dim_y,
            num_layers=num_decoder_layers,
        ),
        learn_σ ? param([log(σ)]) : [log(σ)]
    )
end

"""
    loglik(
        model::AbstractNP,
        epoch::Integer,
        xc,
        yc,
        xt,
        yt;
        num_samples::Integer,
        importance_weighted::Bool=true
    )

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
- `importance_weighted::Bool=true`: Do an importance-weighted estimate.

# Returns
- `Real`: Average negative log-expected likelihood.
"""
function loglik(
    model::AbstractNP,
    epoch::Integer,
    xc,
    yc,
    xt,
    yt;
    num_samples::Integer,
    importance_weighted::Bool=true
)
    if importance_weighted
        # Perform deterministic and latent encoding.
        xz, pz, r = encode(model, xc, yc, xt)

        # Construct posterior over latent variable for an importance-weighted estimate.
        qz = encode_lat(model, cat(xc, xt, dims=1), cat(yc, yt, dims=1), xz)

        # Sample latent variable and perform decoding.
        z = _sample(qz..., num_samples)
        μ, σ = decode(model, xz, z, r, xt), exp.(model.log_σ)

        # Do an importance weighted estimate.
        weights = _logpdf(z, pz...) .- _logpdf(z, qz...)
    else
        # Sample from the prior.
        μ, σ = model(xc, yc, xt, num_samples)

        # Do a regular Monte Carlo estimate.
        weights = 0
    end

    # Perform Monte Carlo estimate.
    logpdfs = weights .+ _logpdf(yt, μ, σ)

    # Log-mean-exp over samples.
    logpdfs = logsumexp(logpdfs, dims=4) .- Float32(log(num_samples))

    # Return average over batches.
    return -mean(logpdfs)
end

_logpdf(xs::AbstractArray...) = sum(gaussian_logpdf(xs...), dims=(1, 2))
_logpdf(ys::Tuple, ds::Tuple...) =
    reduce((x, y) -> x .+ y, [_logpdf(y, d...) for (y, d) in zip(ys, ds)])

"""
    elbo(model::AbstractNP, epoch::Integer, xc, yc, xt, yt, num_samples::Integer)

Neural process ELBO-style loss. Subsumes the context set into the target set.

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
    # We subsume the context set into the target set for this ELBO.
    x_all = cat(xc, xt, dims=1)
    y_all = cat(yc, yt, dims=1)

    # Perform deterministic and latent encoding.
    xz, pz, r = encode(model, xc, yc, x_all)

    # Construct posterior over latent variable.
    qz = encode_lat(model, x_all, y_all, xz)

    # Sample latent variable and perform decoding.
    z = _sample(qz..., num_samples)
    μ, σ = decode(model, xz, z, r, x_all), exp.(model.log_σ)

    # Compute the components of the ELBO.
    exps = _sum(gaussian_logpdf(y_all, μ, σ))
    kls = _sum(kl(qz..., pz...))

    # Estimate ELBO from samples.
    elbos = mean(exps, dims=4) .- kls

    # Return average over batches.
    return -mean(elbos)
end

_sum(x::AbstractArray) = sum(x, dims=(1, 2))  # Sum over data points and channels.
_sum(xs::Tuple) = reduce((x, y) -> x .+ y, _sum.(xs))

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
    μ, σ = untrack(model)(expand_gpu.((xc, yc, xt))..., num_samples)
    samples = μ[:, 1, 1, :] |> cpu
    return nothing, nothing, nothing, samples
end
