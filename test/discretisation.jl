@testset "discretisation.jl" begin
    @testset "UniformDiscretisation1d" begin
        points_per_unit = 64
        margin = 0.1
        multiple = 3

        disc = UniformDiscretisation1d(points_per_unit, margin, multiple)
        x = randn(10, 2, 3)
        x_disc = disc(x)

        # Check repetition of the discretisation.
        @test size(x_disc)[2:3] == (1, 3)

        # Just pick the first one.
        x_disc = x_disc[:, 1, 1]

        # Check that the points per unit is satisfied.
        @test maximum(abs.(x_disc[2:end] - x_disc[1:end - 1])) <= 1 / points_per_unit

        # Check margin.
        @test maximum(x_disc) - maximum(x) >= margin
        @test minimum(x_disc) - minimum(x) <= -margin

        # Check multiple.
        @test mod(length(x_disc), multiple) == 0
    end
end
