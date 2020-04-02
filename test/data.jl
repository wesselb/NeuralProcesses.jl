using Distributions
using Stheno

@testset "data.jl" begin
    @testset "Generate from a GP" begin
        gen = DataGenerator(
            GP(eq(), GPC());
            batch_size=10,
            x_dist=Uniform(-1, 3),
            max_context_points=5,
            num_target_points=8
        )

        epoch = gen(500)  # `num_batches = 500`.

        # Check `num_batches`.
        @test length(epoch) == 500

        # Check minimum numbers of context points.
        @test minimum([size(batch.x_context, 1) for batch in epoch]) == 0
        @test minimum([size(batch.y_context, 1) for batch in epoch]) == 0

        # Check maximum numbers of context points.
        @test maximum([size(batch.x_context, 1) for batch in epoch]) == 5
        @test maximum([size(batch.y_context, 1) for batch in epoch]) == 5

        # Check the numbers of target points.
        @test all([size(batch.x_target, 1) == 8 for batch in epoch])
        @test all([size(batch.y_target, 1) == 8 for batch in epoch])

        for batch in epoch
            # Check `x_dist`.
            for x in [batch.x_context, batch.x_target]
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
            @test size(batch.x_context, 1) == size(batch.y_context, 1)
            @test size(batch.x_target, 1) == size(batch.y_target, 1)
        end
    end
end
