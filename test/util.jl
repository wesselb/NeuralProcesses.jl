using Distributions
using Flux.Tracker

import ConvCNPs:
    ceil_odd, insert_dim, rbf, compute_dists2,
    diagonal, batched_transpose, batched_mul

function test_gradient(f, xs...)
    # Construct scalar version of `f`.
    v = randn(size(f(xs...)))
    f_scalar(ys...) = sum(f(ys...) .* v)

    # Check that non-tracked arguments give a non-tracked output.
    @test !Tracker.istracked(f_scalar(xs...))

    # Compare gradient with numerical estimate.
    grad = Tracker.gradient(f_scalar, xs...)
    grad_estimate = FiniteDifferences.grad(
        central_fdm(5, 1, adapt=1),
        (ys...) -> Tracker.data(f_scalar(ys...)),
        xs...
    )
    @test all(grad .≈ grad_estimate)
end

@testset "util.jl" begin
    @testset "ceil_odd" begin
        @test ceil_odd(2) == 3
        @test ceil_odd(2.5) == 3
        @test ceil_odd(3) == 3
        @test ceil_odd(3.5) == 5
    end

    @testset "insert_dim" begin
        x = randn(2, 3)
        @test size(insert_dim(x, pos=1)) == (1, 2, 3)
        @test size(insert_dim(x, pos=2)) == (2, 1, 3)
        @test size(insert_dim(x, pos=3)) == (2, 3, 1)
    end

    @testset "rbf" begin
        @test rbf([5]) ≈ [exp(-2.5)]
    end

    @testset "compute_dists2" begin
        # Test case of one-dimensional inputs.
        x = randn(3, 1, 2)
        y = randn(5, 1, 2)
        y_perm = permutedims(y, (2, 1, 3))
        @test compute_dists2(x, y) == (x .- y_perm).^2

        # Test case of two-dimensional inputs.
        x = randn(3, 2, 2)
        y = randn(5, 2, 2)
        y_perm = permutedims(y, (2, 1, 3))
        dists2 =
            (x[:, 1:1, :] .- y_perm[1:1, :, :]).^2 .+
            (x[:, 2:2, :] .- y_perm[2:2, :, :]).^2
        @test compute_dists2(x, y) ≈ dists2
    end

    @testset "gaussian_logpdf" begin
        # Test one-dimensional logpdf.
        dist = Normal(1, 2)
        @test logpdf(dist, 3) ≈ gaussian_logpdf([3], [1], [2^2])[1]

        # Test multi-dimensional logpdf.
        function dummy(x, μ, L)
            Σ = L * L' .+ Matrix{Float64}(I, 3, 3)
            return gaussian_logpdf(x, μ, Σ)
        end

        x = randn(3)
        μ = randn(3)
        L = randn(3, 3)
        Σ = L * L' .+ Matrix{Float64}(I, 3, 3)

        @test dummy(x, μ, L) ≈ logpdf(MvNormal(μ, Σ), x) atol=1e-6
        test_gradient(dummy, x, μ, L)
    end

    @testset "diagonal" begin
        x = randn(3)
        @test diagonal(x) ≈ collect(Diagonal(x))
        test_gradient(diagonal, x)
    end

    @testset "batched_transpose" begin
        x = randn(3, 4, 5, 6)
        @test batched_transpose(x) ≈ permutedims(x, (2, 1, 3, 4))
        test_gradient(batched_transpose, x)
    end

    @testset "batched_mul" begin
        x = randn(3, 4, 5, 6)
        y = randn(4, 5, 5, 6)
        z = Array{Float64}(undef, 3, 5, 5, 6)
        for i = 1:5, j = 1:6
            z[:, :, i, j] = x[:, :, i, j] * y[:, :, i, j]
        end
        @test batched_mul(x, y) ≈ z
        test_gradient(batched_mul, x, y)
    end
end
