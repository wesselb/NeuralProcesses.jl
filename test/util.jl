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
    @testset "untrack" begin
        struct MyModel
            θ
        end

        @Flux.treelike MyModel

        model = MyModel(param([1]))

        @test Tracker.istracked(model.θ)
        @test !Tracker.istracked(ConvCNPs.untrack(model).θ)
    end

    @testset "ceil_odd" begin
        @test ConvCNPs.ceil_odd(2) == 3
        @test ConvCNPs.ceil_odd(2.5) == 3
        @test ConvCNPs.ceil_odd(3) == 3
        @test ConvCNPs.ceil_odd(3.5) == 5
    end

    @testset "insert_dim" begin
        x = randn(2, 3)
        @test size(ConvCNPs.insert_dim(x, pos=1)) == (1, 2, 3)
        @test size(ConvCNPs.insert_dim(x, pos=2)) == (2, 1, 3)
        @test size(ConvCNPs.insert_dim(x, pos=3)) == (2, 3, 1)
    end

    @testset "rbf" begin
        @test ConvCNPs.rbf([5]) ≈ [exp(-2.5)]
    end

    @testset "compute_dists²" begin
        # Test case of one-dimensional inputs.
        x = randn(3, 1, 2)
        y = randn(5, 1, 2)
        y_perm = permutedims(y, (2, 1, 3))
        @test ConvCNPs.compute_dists²(x, y) == (x .- y_perm).^2

        # Test case of two-dimensional inputs.
        x = randn(3, 2, 2)
        y = randn(5, 2, 2)
        y_perm = permutedims(y, (2, 1, 3))
        dists² =
            (x[:, 1:1, :] .- y_perm[1:1, :, :]).^2 .+
            (x[:, 2:2, :] .- y_perm[2:2, :, :]).^2
        @test ConvCNPs.compute_dists²(x, y) ≈ dists²
    end

    @testset "gaussian_logpdf" begin
        # Test one-dimensional logpdf.
        dist = Normal(1, 2)
        @test logpdf(dist, 3) ≈ ConvCNPs.gaussian_logpdf([3], [1], [2])[1]

        # Test multi-dimensional logpdf.
        function dummy(x, μ, L)
            Σ = L * L' .+ Matrix{Float64}(I, 3, 3)
            return ConvCNPs.gaussian_logpdf(x, μ, Σ)
        end

        x = randn(3)
        μ = randn(3)
        L = randn(3, 3)
        Σ = L * L' .+ Matrix{Float64}(I, 3, 3)

        @test dummy(x, μ, L) ≈ logpdf(MvNormal(μ, Σ), x) atol=1e-6
        test_gradient(dummy, x, μ, L)
    end

    @testset "kl" begin
        μ₁, σ₁ = [1.2], [0.2]
        μ₂, σ₂ = [2.4], [0.1]

        # Test against a Monte Carlo estimate.
        x = μ₁ .+ σ₁ .* randn(1000000)
        estimate = mean(
            ConvCNPs.gaussian_logpdf(x, μ₁, σ₁) .- ConvCNPs.gaussian_logpdf(x, μ₂, σ₂)
        )
        @test ConvCNPs.kl(μ₁, σ₁,  μ₂, σ₂)[1] ≈ estimate atol=5e-2
    end

    @testset "diagonal" begin
        x = randn(3)
        @test ConvCNPs.diagonal(x) ≈ collect(Diagonal(x))
        test_gradient(ConvCNPs.diagonal, x)
    end

    @testset "batched_transpose" begin
        x = randn(3, 4, 5, 6)
        @test ConvCNPs.batched_transpose(x) ≈ permutedims(x, (2, 1, 3, 4))
        test_gradient(ConvCNPs.batched_transpose, x)
    end

    @testset "batched_mul" begin
        x = randn(3, 4, 5, 6)
        y = randn(4, 5, 5, 6)
        z = Array{Float64}(undef, 3, 5, 5, 6)
        for i = 1:5, j = 1:6
            z[:, :, i, j] = x[:, :, i, j] * y[:, :, i, j]
        end
        @test ConvCNPs.batched_mul(x, y) ≈ z
        test_gradient(ConvCNPs.batched_mul, x, y)
    end

    @testset "logsumexp" begin
        x = randn(3, 4, 5)
        @test StatsFuns.logsumexp(x, dims=1) ≈ ConvCNPs.logsumexp(x, dims=1)
        test_gradient((y) -> ConvCNPs.logsumexp(y, dims=1), x)

        # Test optimisation.
        x = randn(3, 1, 5)
        @test ConvCNPs.logsumexp(x, dims=2) === x
        @test ConvCNPs.logsumexp(x, dims=4) === x
    end

    @testset "softmax" begin
        x = randn(3, 4, 5)
        @test NNlib.softmax(x, dims=1) ≈ ConvCNPs.softmax(x, dims=1)
        test_gradient((y) -> ConvCNPs.softmax(y, dims=1), x)
    end

    @testset "softplus" begin
        x = randn(3, 4, 5)
        @test NNlib.softplus.(x) ≈ ConvCNPs.softplus(x)
    end

    @testset "repeat_cat" begin
        x = randn(3, 1, 2)
        y = randn(1, 5, 2, 4)
        @test ConvCNPs.repeat_cat(x, y, dims=2) ==
            cat(repeat(x, 1, 1, 1, 4), repeat(y, 3, 1, 1, 1), dims=2)

        # Test optimisation.
        @test ConvCNPs.repeat_cat(x, dims=2) === x
    end

    @testset "repeat_gpu" begin
        x = randn(3, 1, 5)
        @test ConvCNPs.repeat_gpu(x, 1, 2, 1, 3) == repeat(x, 1, 2, 1, 3)

        # Test optimisations.
        @test ConvCNPs.repeat_gpu(x, 1, 1) === x
        @test ConvCNPs.repeat_gpu(x, 1, 1, 1, 1) == reshape(x, 3, 1, 5, 1)
    end

    @testset "expand_gpu" begin
        @test size(ConvCNPs.expand_gpu(randn(3))) == (3, 1, 1)
    end

    @testset "slice_at" begin
        x = randn(3, 4, 5)
        @test ConvCNPs.slice_at(x, 2, 2:3) == x[:, 2:3, :]
    end

    @testset "split" begin
        x = randn(3, 4, 5)
        @test ConvCNPs.split(x, 2) == (x[:, 1:2, :], x[:, 3:4, :])
        x = randn(10)
        @test ConvCNPs.split(x) == (x[1:5], x[6:10])
    end

    @testset "split_μ_σ" begin
        μ, σ = ConvCNPs.split_μ_σ(randn(2, 4, 2))
        @test size(μ) == (2, 2, 2)
        @test size(σ) == (2, 2, 2)
        @test all(σ .> 0)
    end

    @testset "with_dummy" begin
        x = randn(3)
        @test ConvCNPs.with_dummy(y -> (@test(size(y) == (3, 1)); y), x) == x
    end

    @testset "to_rank" begin
        # Test compression.
        x = randn(3, 4, 5, 6)
        y, back = ConvCNPs.to_rank(3, x)
        @test y == reshape(x, 3, 4, :)
        @test back(y) == x

        # Test that rank-three tensors are left untouched.
        x = randn(3, 4, 5)
        y, back = ConvCNPs.to_rank(3, x)
        @test x === y
        @test back(x) === x

        # Test that globally broadcasting tensors are left untouched.
        x = randn(1)
        @test back(x) === x
        @test ConvCNPs.to_rank(3, x)[1] === x
    end

    @testset "second" begin
        @test ConvCNPs.second((1, 2, 3)) == 2
    end
end
