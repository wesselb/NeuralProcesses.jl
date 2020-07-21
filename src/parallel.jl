export Parallel

"""
    struct Parallel{N}

A parallel of `N` things.

# Fields
- `xs`: The things that are in parallel.
"""
struct Parallel{N}
    xs
end

Flux.functor(::Type{<:Parallel}, p) = p.xs, xs -> Parallel(xs...)

Parallel(xs...) = Parallel{length(xs)}(collect(xs))

Base.eltype(p::Parallel) = eltype(p.xs)
Base.getindex(p::Parallel, i) = p.xs[i]
Base.iterate(p::Parallel, state...) = iterate(p.xs, state...)
Base.length(p::Parallel) = length(p.xs)
Base.map(f, p::Parallel) = Parallel([map(f, x) for x in p.xs]...)

"""
    flatten(p::Parallel)

Flatten a recursive `Parallel` structure.

# Arguments
- `p::Parallel`: Recursive structure to flatten.

# Returns
- `Vector`: Flattened version of `p`.
"""
flatten(p::Parallel) = vcat(flatten.(p.xs)...)
flatten(x) = [x]

"""
    sample(p::Parallel; num_samples::Integer)

Sample a parallel of distributions.

# Arguments
- `p::Parallel`: Parallel of distributions.

# Keywords
- `num_samples::Integer`: Number of samples to take.

# Returns
- `Parallel`: A parallel of samples.
"""
sample(p::Parallel; num_samples::Integer) =
    Parallel(sample.(p.xs, num_samples=num_samples)...)

"""
    materialise(p::Parallel, f=xs -> repeat_cat(xs..., dims=2))

Turn a parallel of tensors into a tensor.

# Arguments
- `sample::Parallel`: Sample to turn into a tensor.
- `f`: Function that takes in a `Vector` of tensors and turns them into a tensor.

# Returns
- `AA`: Tensor corresponding to parallel.
"""
materialise(p::Parallel, f=xs -> repeat_cat(xs..., dims=2)) = f(materialise.(p.xs, f))
materialise(sample, f) = sample

_add(x, y) = x .+ y

"""
    kl(p::Parallel{N}, q::Parallel{N}) where N

Compute the KL divergence between two parallels of distributions.

# Arguments
- `p::Parallel{N}`: `p`.
- `q::Parallel{N}`: `q`.

# Returns
- `AA`: `KL(p, q)`.
"""
kl(p::Parallel{N}, q::Parallel{N}) where N =
    reduce(_add, [sum(kl(pᵢ, qᵢ), dims=(1, 2)) for (pᵢ, qᵢ) in zip(p, q)])

"""
    logpdf(d::Parallel{N}, x::Parallel{N}) where N

Compute log-pdf of a parallel of distributions at a parallel of values.

# Arguments
- `d::Parallel{N}`: Parallel of distributions.
- `x::Parallel{N}`: Parallel of values.

# Returns
- `AA`: Log-pdfs of `d` as `x`.
"""
logpdf(d::Parallel{N}, x::Parallel{N}) where N =
    reduce(_add, [sum(logpdf(dᵢ, xᵢ), dims=(1, 2)) for (dᵢ, xᵢ) in zip(d, x)])
