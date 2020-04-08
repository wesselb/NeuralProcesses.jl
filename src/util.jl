export gaussian_logpdf

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
    rbf(dist2::AbstractArray)

# Arguments
- `dist2::AbstractArray`: Squared distances.

# Returns
- `AbstractArray`: RBF kernel evaluated at squared distances `dist2`.
"""
rbf(dist2::AbstractArray) = exp.(-0.5f0 .* dist2)

"""
    compute_dists2(x::AbstractArray, y::AbstractArray)

Compute batched pairwise squared distances between 3-tensors `x` and `y`. The batch
dimension is the last dimension.

# Arguments
- `x::T`: Elements that correspond to the rows in the matrices of pairwise distances.
- `y::T`: Elements that correspond to the columns in the matrices of pairwise distances.

# Returns:
- `T`: Pairwise distances between and `x` and `y`.
"""
compute_dists2(x::AbstractArray, y::AbstractArray) = compute_dists2(x, y, Val(size(x, 2)))

compute_dists2(x::AbstractArray, y::AbstractArray, ::Val{1}) =
    (x .- permutedims(y, (2, 1, 3))).^2

function compute_dists2(x::AbstractArray, y::AbstractArray, d::Val)
    y = permutedims(y, (2, 1, 3))
    return sum(x.^2; dims=2) .+ sum(y.^2; dims=1) .- 2 .* batched_mul(x, y)
end

"""
    gaussian_logpdf(x::AbstractArray, μ::AbstractArray, σ²::AbstractArray)

One-dimensional Gaussian log-pdf.

# Arguments
- `x::AbstractArray`: Values to evaluate log-pdf at.
- `μ::AbstractArray`: Means.
- `σ²::AbstractArray`: Variances.

# Returns
- `AbstractArray`: Log-pdf at `x`.
"""
function gaussian_logpdf(x::AbstractArray, μ::AbstractArray, σ²::AbstractArray)
    # Loop fusion was introducing indexing, which severly bottlenecks GPU computation, so
    # we roll out the computation like this.
    logconst = 1.837877f0
    logdet = log.(σ²)
    z = x .- μ
    quad = (z .* z) ./ σ²
    sum = logconst .+ logdet .+ quad
    return -sum ./ 2
end

"""
    gaussian_logpdf(x::AbstractVector, μ::AbstractVector, σ::AbstractArray)

Multi-dimensional Gaussian log-pdf.

# Arguments
- `x::AbstractVector`: Value to evaluate log-pdf at.
- `μ::AbstractVector`: Mean.
- `σ²::AbstractMatrix`: Covariance matrix.

# Returns
- `Real`: Log-pdf at `x`.
"""
gaussian_logpdf(x::AbstractVector, μ::AbstractVector, Σ::AbstractMatrix) =
    Tracker.track(gaussian_logpdf, x, μ, Σ)

@Tracker.grad function gaussian_logpdf(x, μ, Σ)
    n = length(x)
    x, μ, Σ = Tracker.data.((x, μ, Σ))

    U = cholesky(Σ).U  # Upper triangular
    L = U'             # Lower triangular
    z = L \ (x .- μ)
    logconst = 1.837877f0
    # Taking the diagonal of L = U' causes indexing on GPU, which is why we equivalently 
    # take the diagonal of U.
    logpdf = -(n * logconst + 2sum(log.(diag(U))) + dot(z, z)) / 2

    return logpdf, function (ȳ)
        u = U \ z
        eye = gpu(Matrix{Float32}(I, n, n))
        return ȳ .* -u, ȳ .* u, ȳ .* (u .* u' .- U \ (L \ eye)) ./ 2
    end
end

diagonal(x::AbstractVector) = Tracker.track(diagonal, x)

@Tracker.grad function diagonal(x)
    convert(CuArray, Diagonal(Tracker.data(x))), ȳ -> (diag(ȳ),)
end
