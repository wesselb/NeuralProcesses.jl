@testset "discretisation.jl" begin
    @testset "UniformDiscretisation1D" begin
        points_per_unit = 64
        margin = 0.1
        multiple = 3

        disc = UniformDiscretisation1D(points_per_unit, margin, multiple)
        x = randn(10, 2, 3)
        xz = disc(x)

        # Check repetition of the discretisation.
        @test size(xz)[2:3] == (1, 3)

        # Just pick the first one.
        xz = xz[:, 1, 1]

        # Check that the points per unit is satisfied.
        @test maximum(abs.(xz[2:end] - xz[1:end - 1])) <= 1 / points_per_unit

        # Check margin.
        @test maximum(xz) - maximum(x) >= margin
        @test minimum(xz) - minimum(x) <= -margin

        # Check multiple.
        @test mod(length(xz), multiple) == 0
    end
end
