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
    insert_dim(x::T; pos::Integer) where T<:AbstractArray

# Arguments
- `x::T`: Array to insert dimension into.

# Keywords
- `pos::Integer`: Position of the new dimension.

# Returns
- `T`: `x` with an extra dimension at position `pos`.
"""
function insert_dim(x::T; pos::Integer) where T<: AbstractArray
    return reshape(x, size(x)[1:pos - 1]..., 1, size(x)[pos:end]...)
end

"""
    rbf(dist²::AbstractArray)

# Arguments
- `dist²::AbstractArray`: Squared distances.

# Returns
- `AbstractArray`: RBF kernel evaluated at squared distances `dist²`.
"""
rbf(dist²::AbstractArray) = exp.(-0.5f0 .* dist²)

"""
    compute_dists²(x::AbstractArray, y::AbstractArray)

Compute batched pairwise squared distances between 3-tensors `x` and `y`. The batch
dimension is the last dimension.

# Arguments
- `x::T`: Elements that correspond to the rows in the matrices of pairwise distances.
- `y::T`: Elements that correspond to the columns in the matrices of pairwise distances.

# Returns:
- `T`: Pairwise distances between and `x` and `y`.
"""
compute_dists²(x::AbstractArray, y::AbstractArray) = compute_dists²(x, y, Val(size(x, 2)))

compute_dists²(x::AbstractArray, y::AbstractArray, ::Val{1}) =
    (x .- permutedims(y, (2, 1, 3))).^2

function compute_dists²(x::AbstractArray, y::AbstractArray, d::Val)
    y = permutedims(y, (2, 1, 3))
    return sum(x.^2; dims=2) .+ sum(y.^2; dims=1) .- 2 .* batched_mul(x, y)
end

"""
    gaussian_logpdf(x::AbstractArray, μ::AbstractArray, σ::AbstractArray)

One-dimensional Gaussian log-pdf.

# Arguments
- `x::AbstractArray`: Values to evaluate log-pdf at.
- `μ::AbstractArray`: Means.
- `σ::AbstractArray`: Standard deviations.

# Returns
- `AbstractArray`: Log-pdfs at `x`.
"""
function gaussian_logpdf(x::AbstractArray, μ::AbstractArray, σ::AbstractArray)
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
    gaussian_logpdf(x::AbstractVector, μ::AbstractVector, σ::AbstractArray)

Multi-dimensional Gaussian log-pdf.

# Arguments
- `x::AbstractVector`: Value to evaluate log-pdf at.
- `μ::AbstractVector`: Mean.
- `Σ::AbstractMatrix`: Covariance matrix.

# Returns
- `Real`: Log-pdf at `x`.
"""
gaussian_logpdf(x::AbstractVector, μ::AbstractVector, Σ::AbstractMatrix) =
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
    diagonal(x::AbstractVector)

Turn a vector `x` into a diagonal matrix.

# Arguments
- `x::AbstractVector`: Vector.

# Returns
- `AbstractMatrix`: Matrix with `x` on the diagonal.
"""
diagonal(x::AbstractVector) = Tracker.track(diagonal, x)

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
batched_transpose(x::AbstractArray) = Tracker.track(batched_transpose, x)

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
batched_mul(x::AbstractArray, y::AbstractArray) = Tracker.track(batched_mul, x, y)

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
    logsumexp(x::AbstractArray; dims)

Safe log-sum-exp reduction of array `x` along dimensions `dims`.

# Arguments
- `x::AbstractArray`: Array to apply reductions to.
- `dims`: Dimensions along which reduction is applied.

# Returns
- `Real`: Log-sum-exp reduction of `x` along dimensions `dims`.
"""
function logsumexp(x::AbstractArray; dims=:)
    u = maximum(Tracker.data(x), dims=dims)  # Do not track the maximum!
    return u .+ log.(sum(exp.(x .- u), dims=dims))
end

"""
    softmax(x::AbstractArray; dims)

Safe softmax array `x` along dimensions `dims`.

# Arguments
- `x::AbstractArray`: Array to apply softmax to.
- `dims`: Dimensions along which the softmax is applied.

# Returns
- `Real`: Softmax of `x` along dimensions `dims`.
"""
function softmax(x::AbstractArray; dims=:)
    u = maximum(Tracker.data(x), dims=dims)  # Do not track the maximum!
    x = exp.(x .- u)
    return x ./ sum(x, dims=dims)
end

"""
    repeat_gpu(x::AbstractArray, reps...)

Version of `repeat` that has GPU-compatible gradients.

# Arguments
- `x::AbstractArray`: Array to repeat.
- `reps...`: Repetitions.

# Returns
- `AbstractArray`: Repetition of `x`.
"""
repeat_gpu(x::AbstractArray, reps...) = Tracker.track(repeat_gpu, x, reps...)

repeat_gpu(x::CuOrArray, reps...) = repeat(x, reps...)

@Tracker.grad function repeat_gpu(x, reps...)
    # Split into inner and outer repetitions.
    inner = reps[1:min(ndims(x), length(reps))]
    outer = reps[ndims(x) + 1:length(reps)]

    # Check that no complicated repetitions happened.
    for (i, r) in enumerate(inner)
        if r > 1 && size(x, i) != 1
            error("Gradient cannot deal with repetitions of dimensions of size not one.")
        end
    end

    repeat(Tracker.data(x), reps...), function (ȳ)
        # Determine which dimensions to sum and to drop.
        drop_dims = Tuple(ndims(x) + 1:length(reps))
        sum_dims = (findall(r -> r > 1, inner)..., drop_dims...)

        x̄ = ȳ
        length(sum_dims) > 0 && (x̄ = sum(x̄, dims=sum_dims))
        length(drop_dims) > 0 && (x̄ = dropdims(x̄, dims=drop_dims))

        return (x̄, ntuple(_ -> nothing, length(reps))...)
    end
end

"""
    expand_gpu(x::AbstractVector)

Expand a vector to a three-tensor and move it to the GPU.

# Arguments
- `x::AbstractVector`: Vector to expand.

# Returns
- `AbstractArray`: `x` as three-tensor and on the GPU.
"""
expand_gpu(x::AbstractVector) = reshape(x, :, 1, 1) |> gpu

"""
    kl(μ₁, σ₁, μ₂, σ₂)

Kullback--Leibler divergence between one-dimensional Gaussian distributions.

# Arguments
- `μ₁`: Mean of `p`.
- `σ₁`: Standard deviation of `p`.
- `μ₂`: Mean of `q`.
- `σ₂`: Standard deviation of `q`.

# Returns
- `AbstractArray`: `KL(p, q)`.
"""
function kl(μ₁, σ₁, μ₂, σ₂)
    # Loop fusion introduces indexing, which severly bottlenecks GPU computation, so
    # we roll out the computation like this.
    # TODO: What is going on?
    logdet = log.(σ₂ ./ σ₁)
    logdet = 2 .* logdet
    z = μ₁ .- μ₂
    σ₁², σ₂² = σ₁.^2, σ₂.^2
    quad = (σ₁² .+ z .* z) ./ σ₂²
    sum = logdet .+ quad .- 1
    return sum ./ 2
end

"""
    split_μ_σ(channels)

Split a three-tensor into means and standard deviations on dimension two.

# Arguments
- `channels`: Three-tensor to split into means and standard deviations on dimension two.

# Returns
- `Tuple{AbstractArray, AbstractArray}`: Tuple containing means and standard deviations.
"""
function split_μ_σ(channels)
    mod(size(channels, 2), 2) == 0 || error("Number of channels must be even.")
    # Half of the channels are used to determine the mean, and the other half are used to
    # determine the standard deviation.
    i_split = div(size(channels, 2), 2)
    μ = channels[:, 1:i_split, :]
    σ = NNlib.softplus.(channels[:, i_split + 1:end, :])
    return μ, σ
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
    to_rank_3(x::AbstractArray)

Transform `x` into a three-tensor by compression the dimensions `3:end`.

# Arguments
- `x::AbstractArray`: Tensor to compress.

# Returns
- `Tuple`: Tuple containing `x` as a three-tensor and a function to transform back to
    the original dimensions.
"""
function to_rank_3(x::AbstractArray)
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
