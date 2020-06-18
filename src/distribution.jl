"""
    struct Normal

# Fields
- `μ`: Mean.
- `σ`: Standard deviation.
"""
struct Normal
    μ
    σ
end

sample(d::Normal; num_samples::Integer) = d.μ .+ d.σ .* randn_gpu(size(d.μ)..., num_samples)

Statistics.mean(d::Normal) = d.μ
Statistics.std(d::Normal) = d.σ

logpdf(d::Normal, x) = gaussian_logpdf(x, d.μ, d.σ)
kl(p::Normal, q::Normal) = kl(p.μ, p.σ, q.μ, q.σ)

Base.map(d::Normal, f) = Normal(f(d.μ), f(d.σ))

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
Statistics.std(d::Dirac) = zero(d.x)

logpdf(d::Dirac, x) = zero(x)
kl(p::Dirac, q::Dirac) = zero(p.x)

Base.map(d::Dirac, f) = Dirac(f(d.x))
