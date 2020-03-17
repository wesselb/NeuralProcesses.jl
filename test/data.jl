using Distributions
using Stheno

@testset "data.jl" begin
    @testset "Generate from a GP" begin
        gen = DataGenerator(
            eq();
            batch_size=10,
            x_dist=Uniform(0, 2),
            max_context_points=5,
            max_target_points=8
        )

        epoch = gen(500)  # `num_batches = 500`.

        # Check `num_batches`.
        @test length(epoch) == 500

        # Check minimum numbers of context and target points.
        @test minimum([size(batch.x_context, 1) for batch in epoch]) == 3
        @test minimum([size(batch.y_context, 1) for batch in epoch]) == 3
        @test minimum([size(batch.x_target, 1) for batch in epoch]) == 3
        @test minimum([size(batch.y_target, 1) for batch in epoch]) == 3

        # Check maximum numbers of context and target points.
        @test maximum([size(batch.x_context, 1) for batch in epoch]) == 5
        @test maximum([size(batch.y_context, 1) for batch in epoch]) == 5
        @test maximum([size(batch.x_target, 1) for batch in epoch]) == 8
        @test maximum([size(batch.y_target, 1) for batch in epoch]) == 8

        for batch in epoch
            # Check `x_dist`.
            for x in [batch.x_context, batch.x_target]
                @test minimum(x) >= 0
                @test maximum(x) <= 2
            end

            # Check `batch_size`.
            for x in batch
                @test size(x, 3) == 10
            end

            # Check consistency of context and target set.
            @test size(batch.x_context, 1) == size(batch.y_context, 1)
            @test size(batch.x_target, 1) == size(batch.y_target, 1)
        end
    end
end
