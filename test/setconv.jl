function compute_dists²(x::AbstractMatrix, y::AbstractMatrix)
    y = y'
    return sum(x.^2; dims=2) .+ sum(y.^2; dims=1) .- 2 .* (x * y)
end

@testset "setconv.jl" begin
    for density in [true, false]
        scale = 0.1f0
        n_context = 5
        n_target = 10
        dimensionality = 2
        batch_size = 3
        num_channels = 2

        # Generate context and target set.
        xc = randn(Float32, n_context, dimensionality, batch_size)
        yc = randn(Float32, n_context, num_channels, batch_size)
        xt = randn(Float32, n_target, dimensionality, batch_size)

        # Compute with layer.
        layer = set_conv(num_channels, scale; density=density)
        _, out = NeuralProcesses.code(layer, xc, yc, xt)

        # Brute-force the calculation.
        batches = []
        for i in 1:batch_size
            channels = []

            # Compute weights.
            dists² = compute_dists²(xt[:, :, i], xc[:, :, i]) ./ scale.^2
            weights = NeuralProcesses.rbf(dists²)

            # Prepend density channel.
            density && push!(channels, weights * ones(Float32, n_context))

            # Compute other channels.
            for j in 1:num_channels
                channel = weights * yc[:, j, i]
                # Normalise by density channel.
                density && (channel ./= channels[1] .+ 1f-8)
                push!(channels, channel)
            end

            push!(batches, hcat(channels...))
        end
        ref = cat(batches..., dims=3)

        # Check that the brute-force calculation lines up with the layer.
        @test out ≈ ref
    end
end
