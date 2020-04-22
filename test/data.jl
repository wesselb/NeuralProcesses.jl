using Distributions
using Stheno

function test_generator(process)
    gen = DataGenerator(
        process,
        batch_size=10,
        x_dist=Uniform(-1, 3),
        max_context_points=5,
        num_target_points=8
    )

    epoch = gen(500)  # `num_batches = 500`.

    # Check `num_batches`.
    @test length(epoch) == 500

    # Check minimum numbers of context points.
    @test minimum([size(batch[1], 1) for batch in epoch]) == 0
    @test minimum([size(batch[2], 1) for batch in epoch]) == 0

    # Check maximum numbers of context points.
    @test maximum([size(batch[1], 1) for batch in epoch]) == 5
    @test maximum([size(batch[2], 1) for batch in epoch]) == 5

    # Check the numbers of target points.
    @test all([size(batch[3], 1) == 8 for batch in epoch])
    @test all([size(batch[4], 1) == 8 for batch in epoch])

    for batch in epoch
        # Check `x_dist`.
        for x in [batch[1], batch[3]]
            if !isempty(x)
                @test minimum(x) >= -1
                @test maximum(x) <= 3
            end
        end

        # Check `batch_size`.
        for x in batch
            @test size(x, 3) == 10
        end

        # Check consistency of context and target set.
        @test size(batch[1], 1) == size(batch[2], 1)
        @test size(batch[3], 1) == size(batch[4], 1)
    end
end

@testset "data.jl" begin
    @testset "Generate from sawtooth" begin
        test_generator(Sawtooth())
    end
    @testset "Generate from a Bayesian ConvNP" begin
        test_generator(BayesianConvNP())
    end
end
