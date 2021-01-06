"""
    struct Gaussian

# Fields
- `μ`: Mean.
- `σ`: Standard deviation.
"""
struct Gaussian
    μ
    σ
end

sample(d::Gaussian; num_samples::Integer) =
    d.μ .+ d.σ .* randn_gpu(data_eltype(d.μ), size(d.μ)..., num_samples)

Statistics.mean(d::Gaussian) = d.μ
Statistics.std(d::Gaussian) = d.σ

logpdf(d::Gaussian, x) = gaussian_logpdf(x, d.μ, d.σ)
kl(p::Gaussian, q::Gaussian) = kl(p.μ, p.σ, q.μ, q.σ)

Base.map(f, d::Gaussian) = Gaussian(f(d.μ), f(d.σ))

"""
    struct Dirac

The log-pdf of a `Dirac` is defined as `0` everywhere, and the KL divergence between
any two `Dirac`s is also defined as `0`.

# Fields
- `x`: Position.
"""

struct Dirac
    x
end

sample(d::Dirac; num_samples::Integer) = d.x

Statistics.mean(d::Dirac) = d.x
Statistics.std(d::Dirac) = 0

logpdf(d::Dirac, x) = 0
kl(p::Dirac, q::Dirac) = 0

Base.map(f, d::Dirac) = Dirac(f(d.x))

# We need to help `sum` to work with scalars and the keyword argument `dims`.
Base.sum(x::Real; dims=nothing) = x

"""
    struct MultivariateGaussian

# Fields
- `μ`: Mean.
- `Σ`: Covariance matrix.
"""
struct MultivariateGaussian
    μ
    Σ
end

function sample(d::MultivariateGaussian; num_samples::Integer)
    n, num_channels, batch_size = size(d.μ)

    μ = d.μ
    Σ = d.Σ

    samples_channels = []
    for c = 1:num_channels
        samples_batches = []
        for b = 1:batch_size
            ε = cholesky(Σ[:, :, c, b]).U' * randn_gpu(data_eltype(d.μ), n, num_samples)
            push!(samples_batches, μ[:, c:c, b:b] .+ reshape(ε, n, 1, 1, num_samples))
        end
        push!(samples_channels, cat(samples_batches..., dims=3))
    end
    sample = cat(samples_channels..., dims=2)

    return sample
end

Statistics.mean(d::MultivariateGaussian) = d.μ
function Statistics.std(d::MultivariateGaussian)
    Σ = d.Σ
    n = size(Σ, 1)
    return sqrt.(cat([Σ[i, i:i, :, :] for i = 1:n]..., dims=1))
end
Statistics.var(d::MultivariateGaussian) = d.Σ

function logpdf(d::MultivariateGaussian, x)
    _, num_channels, batch_size = size(x)

    μ = d.μ
    Σ = d.Σ

    logpdfs_channels = []
    for c = 1:num_channels
        logpdfs_batches = []
        for b = 1:batch_size
            push!(logpdfs_batches, gaussian_logpdf(x[:, c, b], μ[:, c, b], Σ[:, :, c, b]))
        end
        push!(logpdfs_channels, cat(logpdfs_batches..., dims=3))
    end
    logpdfs = cat(logpdfs_channels..., dims=2)

    return logpdfs
end

Base.map(f, d::MultivariateGaussian) = Gaussian(f(d.μ), f(d.Σ))
