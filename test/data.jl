function test_generator(process)
    gen = DataGenerator(
        process,
        batch_size=10,
        x_context=Uniform(-1, 3),
        x_target=Uniform(5, 10),
        num_context=DiscreteUniform(0, 5),
        num_target=DiscreteUniform(8, 8),
        σ²=1e-2
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
        # Check `x_context`.
        if !isempty(batch[1])
            @test minimum(batch[1]) >= -1
            @test maximum(batch[1]) <= 3
        end

        # Check `x_target`.
        if !isempty(batch[3])
            @test minimum(batch[3]) >= 5
            @test maximum(batch[3]) <= 10
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
    @testset "UniformUnion" begin
        n = 10000
        dist = UniformUnion(Uniform(1, 2), Uniform(3, 5))
        x = rand(dist, n)

        # Check that the probabilities are calculated correctly.
        @test sum(x .<= 2) ≈ n / 3 rtol=5e-2
        @test sum(x .>= 3) ≈ 2n / 3 rtol=5e-2
    end
    @testset "Generate from sawtooth" begin
        test_generator(Sawtooth())
    end
    @testset "Generate from a Bayesian ConvNP" begin
        test_generator(BayesianConvNP())
    end
    @testset "Generate from a mixture" begin
        test_generator(Mixture(Sawtooth(), BayesianConvNP()))
    end
end
