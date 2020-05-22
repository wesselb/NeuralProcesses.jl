export gaussian_logpdf, logsumexp

"""
    untrack(model)

Untrack a model in Flux.

# Arguments
- `model`: Model to untrack.

# Returns
- Untracked model.
"""
untrack(model) = mapleaves(x -> Flux.data(x), model)

"""
    ceil_odd(x::T) where T<:Real

Ceil a number to the nearest odd integer.

# Arguments
- `x::T`: Number to ceil.

# Returns
- `Integer`: Nearest odd integer equal to or above `x`.
"""
ceil_odd(x::T) where T<:Real = Integer(ceil((x - 1) / 2) * 2 + 1)

"""
    insert_dim(x::T; pos::Integer) where T<:AA

# Arguments
- `x::T`: Array to insert dimension into.

# Keywords
- `pos::Integer`: Position of the new dimension.

# Returns
- `T`: `x` with an extra dimension at position `pos`.
"""
function insert_dim(x::T; pos::Integer) where T<: AA
    return reshape(x, size(x)[1:pos - 1]..., 1, size(x)[pos:end]...)
end

"""
    rbf(dist²::AA)

# Arguments
- `dist²::AA`: Squared distances.

# Returns
- `AA`: RBF kernel evaluated at squared distances `dist²`.
"""
rbf(dist²::AA) = exp.(-0.5f0 .* dist²)

"""
    compute_dists²(x::AA, y::AA)

Compute batched pairwise squared distances between 3-tensors `x` and `y`. The batch
dimension is the last dimension.

# Arguments
- `x::T`: Elements that correspond to the rows in the matrices of pairwise distances.
- `y::T`: Elements that correspond to the columns in the matrices of pairwise distances.

# Returns:
- `T`: Pairwise distances between and `x` and `y`.
"""
compute_dists²(x::AA, y::AA) = compute_dists²(x, y, Val(size(x, 2)))

compute_dists²(x::AA, y::AA, ::Val{1}) =
    (x .- permutedims(y, (2, 1, 3))).^2

function compute_dists²(x::AA, y::AA, d::Val)
    y = permutedims(y, (2, 1, 3))
    return sum(x.^2; dims=2) .+ sum(y.^2; dims=1) .- 2 .* batched_mul(x, y)
end

"""
    gaussian_logpdf(x::AA, μ::AA, σ::AA)

One-dimensional Gaussian log-pdf.

# Arguments
- `x::AA`: Values to evaluate log-pdf at.
- `μ::AA`: Means.
- `σ::AA`: Standard deviations.

# Returns
- `AA`: Log-pdfs at `x`.
"""
function gaussian_logpdf(x::AA, μ::AA, σ::AA)
    # Loop fusion introduces indexing, which severly bottlenecks GPU computation, so
    # we roll out the computation like this.
    # TODO: What is going on?
    logconst = 1.837877f0
    logdet = 2 .* log.(σ)
    z = (x .- μ) ./ σ
    quad = z .* z
    sum = logconst .+ logdet .+ quad
    return -sum ./ 2
end

"""
    gaussian_logpdf(x::AV, μ::AV, σ::AA)

Multi-dimensional Gaussian log-pdf.

# Arguments
- `x::AV`: Value to evaluate log-pdf at.
- `μ::AV`: Mean.
- `Σ::AM`: Covariance matrix.

# Returns
- `Real`: Log-pdf at `x`.
"""
gaussian_logpdf(x::AV, μ::AV, Σ::AM) =
    Tracker.track(gaussian_logpdf, x, μ, Σ)

function _gaussian_logpdf(x, μ, Σ)
    n = length(x)

    U = cholesky(Σ).U  # Upper triangular
    L = U'             # Lower triangular
    z = L \ (x .- μ)
    logconst = 1.837877f0
    # Taking the diagonal of L = U' causes indexing on GPU, which is why we equivalently
    # take the diagonal of U.
    logpdf = -(n * logconst + 2sum(log.(diag(U))) + dot(z, z)) / 2

    return logpdf, n, L, U, z
end

gaussian_logpdf(x::CuOrVector, μ::CuOrVector, Σ::CuOrMatrix) =
    first(_gaussian_logpdf(x, μ, Σ))

@Tracker.grad function gaussian_logpdf(x, μ, Σ)
    logpdf, n, L, U, z = _gaussian_logpdf(Tracker.data.((x, μ, Σ))...)
    return logpdf, function (ȳ)
        u = U \ z
        eye = gpu(Matrix{Float32}(I, n, n))
        return ȳ .* -u, ȳ .* u, ȳ .* (u .* u' .- U \ (L \ eye)) ./ 2
    end
end

"""
    diagonal(x::AV)

Turn a vector `x` into a diagonal matrix.

# Arguments
- `x::AV`: Vector.

# Returns
- `AM`: Matrix with `x` on the diagonal.
"""
diagonal(x::AV) = Tracker.track(diagonal, x)

diagonal(x::Array{T, 1}) where T<:Real = convert(Array, Diagonal(x))

@Tracker.grad function diagonal(x)
    return diagonal(Tracker.data(x)), ȳ -> (diag(ȳ),)
end

"""
    batched_transpose(x)

Batch transpose tensor `x` where dimensions `1:2` are the matrix dimensions and dimension
`3:end` are the batch dimensions.

# Arguments
- `x`: Tensor to transpose.

# Returns
- Transpose of `x`.
"""
batched_transpose(x::AA) = Tracker.track(batched_transpose, x)

batched_transpose(x::CuOrArray) =
    permutedims(x, (2, 1, range(3, length(size(x)), step=1)...))

@Tracker.grad function batched_transpose(x)
    return batched_transpose(Tracker.data(x)), ȳ -> (batched_transpose(ȳ),)
end

"""
    batched_mul(x, y)

Batch matrix-multiply tensors `x` and `y` where dimensions `1:2` are the matrix
dimensions and dimension `3:end` are the batch dimensions.

# Args
- `x`: Left matrix in product.
- `y`: Right matrix in product.

# Returns
- Matrix product of `x` and `y`.
"""
batched_mul(x::AA, y::AA) = Tracker.track(batched_mul, x, y)

function _batched_mul(x, y)
    x, back = to_rank_3(x)
    y, _ = to_rank_3(y)
    return back(Flux.batched_mul(x, y)), x, y
end

batched_mul(x::CuOrArray, y::CuOrArray) = first(_batched_mul(x, y))

@Tracker.grad function batched_mul(x, y)
    z, x, y = _batched_mul(Tracker.data.((x, y))...)
    return z, function (ȳ)
        ȳ, back = to_rank_3(ȳ)
        return (
            back(Flux.batched_mul(ȳ, batched_transpose(y))),
            back(Flux.batched_mul(batched_transpose(x), ȳ))
        )
    end
end

"""
    logsumexp(x::AA; dims)

Safe log-sum-exp reduction of array `x` along dimensions `dims`.

# Arguments
- `x::AA`: Array to apply reductions to.
- `dims`: Dimensions along which reduction is applied.

# Returns
- `Real`: Log-sum-exp reduction of `x` along dimensions `dims`.
"""
function logsumexp(x::AA; dims=:)
    u = maximum(Tracker.data(x), dims=dims)  # Do not track the maximum!
    return u .+ log.(sum(exp.(x .- u), dims=dims))
end

"""
    softmax(x::AA; dims)

Safe softmax array `x` along dimensions `dims`.

# Arguments
- `x::AA`: Array to apply softmax to.
- `dims`: Dimensions along which the softmax is applied.

# Returns
- `Real`: Softmax of `x` along dimensions `dims`.
"""
function softmax(x::AA; dims=:)
    u = maximum(Tracker.data(x), dims=dims)  # Do not track the maximum!
    x = exp.(x .- u)
    return x ./ sum(x, dims=dims)
end

function softplus(x::AA)
    return log.(1 .+ exp.(-abs.(x))) .+ max.(x, 0)
end

function repeat_cat(xs...; dims)
    # Determine the maximum rank.
    max_rank = maximum(ndims.(xs))

    # Get the sizes of the inputs, extending to the maximum rank.
    sizes = [(size(x)..., ntuple(_ -> 1, max_rank - ndims(x))...) for x in xs]

    # Determine the maximum size of each dimension.
    max_size = maximum.(zip(sizes...))

    # Determine the repetitions for each input.
    reps = [div.(max_size, s) for s in sizes]

    # Do not repeat along the concatenation dimension.
    for i in 1:length(reps)
        reps[i][dims] = 1
    end

    # Repeat every element appropriately many times.
    xs = [repeat_gpu(x, r...) for (x, r) in zip(xs, reps)]

    # Return concatenation.
    return cat(xs..., dims=dims)
end

repeat_gpu(x, reps...) = x .* ones_gpu(Float32, reps...)

"""
    expand_gpu(x::AV)

Expand a vector to a three-tensor and move it to the GPU.

# Arguments
- `x::AV`: Vector to expand.

# Returns
- `AA`: `x` as three-tensor and on the GPU.
"""
expand_gpu(x::AV) = reshape(x, :, 1, 1) |> gpu

"""
    kl(μ₁::AA, σ₁::AA, μ₂::AA, σ₂::AA)

Kullback--Leibler divergence between one-dimensional Gaussian distributions.

# Arguments
- `μ₁::AA`: Mean of `p`.
- `σ₁::AA`: Standard deviation of `p`.
- `μ₂::AA`: Mean of `q`.
- `σ₂::AA`: Standard deviation of `q`.

# Returns
- `AA`: `KL(p, q)`.
"""
function kl(μ₁::AA, σ₁::AA, μ₂::AA, σ₂::AA)
    # Loop fusion introduces indexing, which severly bottlenecks GPU computation, so
    # we roll out the computation like this.
    # TODO: What is going on?
    logdet = log.(σ₂ ./ σ₁)
    logdet = 2 .* logdet  # This cannot be combined with the `log`.
    z = μ₁ .- μ₂
    σ₁², σ₂² = σ₁.^2, σ₂.^2  # This must be separated from the calulation in `quad`.
    quad = (σ₁² .+ z .* z) ./ σ₂²
    sum = logdet .+ quad .- 1
    return sum ./ 2
end

"""
    kl(p₁::Tuple, p₂::Tuple, q₁::Tuple, q₂::Tuple)

Kullback--Leibler divergences between multiple one-dimensional Gaussian distributions.

# Arguments
- `p₁::Tuple`: Means and standard deviations corresponding to `p₁`.
- `p₂::Tuple`: Means and standard deviations corresponding to `p₂`.
- `q₁::Tuple`: Means and standard deviations corresponding to `q₁`.
- `q₂::Tuple`: Means and standard deviations corresponding to `q₂`.

# Returns
- `Tuple{AA, AA}`: `KL(p₁, q₁)` and `KL(p₂, q₂)`.
"""
kl(p₁::Tuple, p₂::Tuple, q₁::Tuple, q₂::Tuple) = kl(p₁..., q₁...), kl(p₂..., q₂...)

slice_at(x, i, slice) = getindex(x, ntuple(j -> i == j ? slice : Colon(), ndims(x))...)

function split(x, dim)
    mod(size(x, dim), 2) == 0 || error("Size of dimension $dim must be even.")
    i = div(size(x, dim), 2)  # Determine index at which to split.
    return slice_at(x, dim, 1:i), slice_at(x, dim, i + 1:size(x, dim))
end

function split(x)
    mod(length(x), 2) == 0 || error("Length of input must be even")
    i = div(length(x), 2)  # Determine index at which to split.
    return x[1:i], x[i + 1:end]
end

"""
    split_μ_σ(channels)

Split a three-tensor into means and standard deviations on dimension two.

# Arguments
- `channels`: Three-tensor to split into means and standard deviations on dimension two.

# Returns
- `Tuple{AA, AA}`: Tuple containing means and standard deviations.
"""
function split_μ_σ(channels)
    μ, transformed_σ = split(channels, 2)
    return μ, softplus(transformed_σ)
end

"""
    with_dummy(f, x)

Insert dimension two to `x` before applying `f` and remove dimension two afterwards.

# Arguments
- `f`: Function to apply.
- `x`: Input to `f`.

# Returns
- `f(x)`.
"""
with_dummy(f, x) = dropdims(f(insert_dim(x, pos=2)), dims=2)

"""
    to_rank_3(x::AA)

Transform `x` into a three-tensor by compression the dimensions `3:end`.

# Arguments
- `x::AA`: Tensor to compress.

# Returns
- `Tuple`: Tuple containing `x` as a three-tensor and a function to transform back to
    the original dimensions.
"""
function to_rank_3(x::AA)
    # If `x` is already rank three, there is nothing to be done.
    if ndims(x) == 3
        return x, identity
    end

    # Reshape `x` into a three-tensor.
    size_x = size(x)
    return reshape(x, size_x[1:2]..., prod(size_x[3:end])), function (y)
        return reshape(y, size(y)[1:2]..., size_x[3:end]...)
    end
end
