import ConvCNPs: rbf

function compute_dists2(x::AbstractMatrix, y::AbstractMatrix)
    y = y'
    return sum(x.^2; dims=2) .+ sum(y.^2; dims=1) .- 2 .* (x * y)
end


@testset "setconv.jl" begin
    for density in [true, false]
        scale = 0.1
        n_context = 5
        n_target = 10
        batch_size = 3
        num_channels = 2

        layer = set_conv(2, scale; density=density)

        # Generate a context and target set.
        x_context = randn(n_context, 2, batch_size)
        if density
            y_context = randn(n_context, num_channels, batch_size)
        else
            y_context = randn(n_context, num_channels, batch_size)
        end
        x_target = randn(n_target, 2, batch_size)

        # Brute-force the calculation.
        batches = []
        for i in 1:batch_size
            dists2 = compute_dists2(x_target[:, :, i], x_context[:, :, i]) ./ scale.^2
            weights = rbf.(dists2)

            channels = []
            if density
                push!(channels, weights * ones(n_context))
            end
            for j in 1:num_channels
                channel = weights * y_context[:, j, i]
                if density
                    channel ./= channels[1] .+ 1e-8
                end
                push!(channels, channel)
            end
            
            push!(batches, hcat(channels...))
        end
        y_target_reference = cat(batches...; dims=3)

        # Check that the brute-force calculation lines up with the layer.
        @test layer(x_context, y_context, x_target) ≈ y_target_reference
    end
end
