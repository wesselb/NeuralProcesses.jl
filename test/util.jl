using Distributions

import ConvCNPs: ceil_odd, insert_dim, rbf, compute_dists2

@testset "util.jl" begin
    @testset "ceil_odd" begin
        @test ceil_odd(2) == 3
        @test ceil_odd(2.5) == 3
        @test ceil_odd(3) == 3
        @test ceil_odd(3.5) == 5
    end

    @testset "insert_dim" begin
        x = randn(2, 3)
        @test size(insert_dim(x; pos=1)) == (1, 2, 3)
        @test size(insert_dim(x; pos=2)) == (2, 1, 3)
        @test size(insert_dim(x; pos=3)) == (2, 3, 1)
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
        dist = Normal(1, 2)
        @test logpdf(dist, 3) ≈ gaussian_logpdf([3], [1], [2])[1]
    end
end
