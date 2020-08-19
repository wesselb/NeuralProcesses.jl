if CUDA.functional()
    _cuda_log = CUDA.log
else
    _cuda_log = log
end

"""
    Fixed

Wrap a thing to indicate that it must not be learned.

# Fields
- `x`: Thing that must not be learned.
"""
struct Fixed
    x
end

# Make `Fixed` compatible with Flux's mechanisms of moving things to and from the GPU.
CUDA.cu(x::Fixed) = Fixed(CUDA.cu(x.x))
Flux.adapt_storage(T::Type{<:Array}, x::Fixed) = Fixed(Flux.adapt(T, x.x))

"""
    unwrap(x::Fixed)

Unwrap a thing wrapped by `Fixed`.

# Arguments
- `x::Fixed`: Wrapped thing.

# Returns
- Unwrapped thing.
"""
unwrap(x::Fixed) = x.x
unwrap(x) = x

"""
    untrack(model)

Untrack a model in Tracker.

# Arguments
- `model`: Model to untrack.

# Returns
- Untracked model.
"""
untrack(model) = fmap(x -> Tracker.data(x), model)

"""
    track(model)

Track a model with Tracker.

Any `AbstractArray{<:Number}` is tracked. Everything else is not tracked. This is similar
to how `Flux.params` operates.

# Arguments
- `model`: Model to track.

# Returns
- Tracked model.
"""
track(model) = fmap(_track, model)
_track(x) = x
_track(x::AbstractArray{<:Number}) = Tracker.param(x)

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
function insert_dim(x::T; pos::Integer) where T<:AA
    return reshape(x, size(x)[1:pos - 1]..., 1, size(x)[pos:end]...)
end

"""
    rbf(dists²::AA)

# Arguments
- `dists²::AA`: Squared distances.

# Returns
- `AA`: RBF kernel evaluated at squared distances `dists²`.
"""
rbf(dists²::AA) = exp.(-0.5f0 .* dists²)

"""
    compute_dists²(x::AA, y::AA)

Compute batched pairwise squared distances between tensors `x` and `y`. The batch
dimensions are the third to last.

# Arguments
- `x::AA`: Elements that correspond to the rows in the matrices of pairwise distances.
- `y::AA`: Elements that correspond to the columns in the matrices of pairwise distances.

# Returns:
- `AA`: Pairwise distances between and `x` and `y`.
"""
compute_dists²(x::AA, y::AA) = compute_dists²(x, y, Val(size(x, 2)))

compute_dists²(x::AA, y::AA, ::Val{1}) = (x .- batched_transpose(y)).^2

function compute_dists²(x::AA, y::AA, d::Val)
    y = batched_transpose(y)
    return sum(x.^2, dims=2) .+ sum(y.^2, dims=1) .- 2 .* batched_mul(x, y)
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
    # We unroll the computation to avoid loop fusion, which causes GPU issues.
    logconst = 1.837877f0
    logdet = _cuda_log.(σ)
    logdet = 2 .* logdet
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
    logpdf = -(n * logconst + 2sum(_cuda_log.(diag(U))) + dot(z, z)) / 2

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
    # We unroll the computation to avoid loop fusion, which causes GPU issues.
    logdet = _cuda_log.(σ₂ ./ σ₁)
    logdet = 2 .* logdet
    z = μ₁ .- μ₂
    σ₁², σ₂² = σ₁.^2, σ₂.^2
    quad = (σ₁² .+ z .* z) ./ σ₂²
    sum = logdet .+ quad .- 1
    return sum ./ 2
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
    permutedims(x, (2, 1, 3:ndims(x)...))

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
    x, back = to_rank(3, x)
    y, _ = to_rank(3, y)
    return back(Flux.batched_mul(x, y)), x, y
end

batched_mul(x::CuOrArray, y::CuOrArray) = first(_batched_mul(x, y))

@Tracker.grad function batched_mul(x, y)
    z, x, y = _batched_mul(Tracker.data.((x, y))...)
    return z, function (ȳ)
        ȳ, back = to_rank(3, ȳ)
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
- `AA`: Log-sum-exp reduction of `x` along dimensions `dims`.
"""
function logsumexp(x::AA; dims=:)
    # Only do work if there is work to be done.
    !_must_work(x, dims) && (return x)
    u = maximum(Tracker.data(x), dims=dims)  # Do not track the maximum!
    # We unroll the computation to avoid loop fusion, which causes GPU issues.
    z = sum(exp.(x .- u), dims=dims)
    z = _cuda_log.(z)
    return u .+ z
end

function _must_work(x, dims)
    if dims == Colon()
        # We operate on all dimensions.
        return length(x) > 1
    else
        # We operate on a subset of the dimensions.
        return any([size(x, d) > 1 for d in dims])
    end
end

"""
    softmax(x::AA; dims)

Safe softmax array `x` along dimensions `dims`.

# Arguments
- `x::AA`: Array to apply softmax to.
- `dims`: Dimensions along which the softmax is applied.

# Returns
- `AA`: Softmax of `x` along dimensions `dims`.
"""
function softmax(x::AA; dims=:)
    u = maximum(Tracker.data(x), dims=dims)  # Do not track the maximum!
    x = exp.(x .- u)
    return x ./ sum(x, dims=dims)
end

"""
    softplus(x::AA)

Safe softplus.

# Arguments
- `x::AA`: Array to apply softplus to.

# Returns
- `AA`: Softplus applied to every element in `x`.
"""
function softplus(x::AA)
    # We unroll the computation to avoid loop fusion, which causes GPU issues.
    z = 1 .+ exp.(-abs.(x))
    z = _cuda_log.(z)
    return z .+ max.(x, 0)
end

"""
    repeat_cat(xs::MaybeAA...; dims::Integer)

Repeat the tensors `xs` appropriately many times to concatenate them along dimension `dims`.

# Arguments
- `xs::MaybeAA...`: Tensors to concatenate.
- `dims::Integer`: Dimensions to concatenate along.

# Returns
- `AA`: Concatenation of `xs`.
"""
repeat_cat(xs::AA...; dims::Integer) = cat(_repeat(xs...; dims=dims)..., dims=dims)
repeat_cat(xs::MaybeAA...; dims::Integer) =
    repeat_cat(_filter_nothings(xs)...; dims=dims)
repeat_cat(x::AA; dims::Integer) = x

"""
    repeat_merge(xs::MaybeAA...; dims::Integer)

Repeat the tensors `xs` appropriately many times to merge them along dimension `dims`.

# Arguments
- `xs::MaybeAA...`: Tensors to merge.
- `dims::Integer`: Dimensions to merge along.

# Returns
- `AA`: Merge of `xs`.
"""
function repeat_merge(xs::AA...; dims::Integer)
    xs = _repeat(xs...; dims=dims)

    # We can only merge if all inputs are equal.
    !all([xs[1] ≈ x for x in xs[2:end]]) && error("Not all inputs are approximately equal.")

    # Since all inputs are equal, their merge is simply any, so take the first one.
    return xs[1]
end
repeat_merge(xs::MaybeAA...; dims::Integer) =
    repeat_merge(_filter_nothings(xs)...; dims=dims)
repeat_merge(x::AA; dims::Integer) = x


_filter_nothings(xs) = filter(x -> !isnothing(x), collect(xs))

function _repeat(xs::MaybeAA...; dims::Integer)
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

    # Repeat every element appropriately many times and return.
    return [repeat_gpu(x, r...) for (x, r) in zip(xs, reps)]
end


"""
    repeat_gpu(x::AA, reps::Integer...)

Repeat that is compatible with the GPU. This is not a full substitute of `repeat`: it can
only repeat along dimensions of size one.

# Arguments
- `x::AA`: Tensor to repeat.
- `reps::Integer...`: Repetitions.

# Returns
- `AA`: `x` repeated `reps` times along every dimension.
"""
function repeat_gpu(x::AA, reps::Integer...)
    # Only do work if there is work to be done.
    all(reps .== 1) && ndims(x) >= length(reps) && (return x)
    all(reps .== 1) && (return reshape(
        x, size(x)..., ntuple(_ -> 1, length(reps) - ndims(x))...
    ))
    return x .* ones_gpu(Float32, reps...)
end

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
    slice_at(x::AA, dim::Integer, slice)

Slice `x` at dimension `dim` with slice `slice`.

# Arguments
- `x::AA`: Tensor to slice.
- `dim::Integer`: Dimension to slice.
- `slice`: Slice to get.

# Returns
- `AA`: `x[..., slice, ...]` with `slice` at position `dim`.
"""
slice_at(x::AA, dim::Integer, slice) =
    getindex(x, ntuple(i -> i == dim ? slice : Colon(), ndims(x))...)

"""
    split(x, dim::Integer)
    split(x)

Split `x` into two.

# Arguments
- `x`: Thing to split into two.
- `dim::Integer`: Dimension to split along.

# Returns
- `Tuple`: Two halves of `x`.
"""
function split(x, dim::Integer)
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
    with_dummy(f)

Insert dimension two to `x` before applying `f` and remove dimension two afterwards.

# Arguments
- `f`: Function to apply.
- `x`: Input to `f`.

# Returns
- `f(x)`.
"""
with_dummy(f, x) = dropdims(f(insert_dim(x, pos=2)), dims=2)

"""
    to_rank(rank, x::AA)

Transform `x` into a tensor of rank `rank` by compressing the dimensions `rank + 1:end`.

# Arguments
- `rank::Integer`: Desired rank.
- `x::AA`: Tensor to compress.

# Returns
- `Tuple`: Tuple containing `x` as a `rank`-tensor and a function to transform back to
    the original dimensions.
"""
function to_rank(rank::Integer, x::AA)
    # If `x` is already rank `rank`, there is nothing to be done.
    ndims(x) == rank && (return x, identity)

    # If `x` is a tensor that broadcasts to anything, do nothing.
    size(x) == (1,) && (return x, identity)

    # Reshape `x` into a `rank`-tensor.
    size_x = size(x)
    return reshape(x, size_x[1:rank - 1]..., prod(size_x[rank:end])), function (y)
        size(y) == (1,) && (return y)
        return reshape(y, size(y)[1:rank - 1]..., size_x[rank:end]...)
    end
end

"""
    second(x)

Get the second element of `x`. This function complements `first`.

# Arguments
- `x`: Object to get second element of.

# Returns
- Second element of `x`.
"""
second(x) = x[2]


"""
    data_eltype(x)

Get the element type of the data of `x`. This removes any tracking from Tracker.

# Arguments
- `x`: Container to get element type of.

# Returns
- Type of the elements of `x` with all tracking removed.
"""
data_eltype(x) = eltype(Tracker.data(x))
