import ConvCNPs: rbf, insert_dim

function compute_dists2(x::AbstractMatrix, y::AbstractMatrix)
    y = y'
    return sum(x.^2; dims=2) .+ sum(y.^2; dims=1) .- 2 .* (x * y)
end

@testset "setconv.jl" begin
    for perform_encoding in [true, false]
        scale = 0.1f0
        n_context = 5
        n_target = 10
        dimensionality = 2
        batch_size = 3
        num_channels = 2

        # Generate context and target set.
        x_context = randn(Float32, n_context, dimensionality, batch_size)
        y_context = randn(Float32, n_context, num_channels, batch_size)
        x_target = randn(Float32, n_target, dimensionality, batch_size)

        # Compute with layer.
        layer = set_conv(num_channels + perform_encoding, scale)
        if perform_encoding
            out = encode(layer, x_context, y_context, x_target)
        else
            out = decode(layer, x_context, insert_dim(y_context, pos=2), x_target)
        end

        # Brute-force the calculation.
        batches = []
        for i in 1:batch_size
            channels = []

            # Compute weights.
            dists2 = compute_dists2(x_target[:, :, i], x_context[:, :, i]) ./ scale.^2
            weights = rbf(dists2)

            if perform_encoding
                # Prepend density channel only for the encoding.
                push!(channels, weights * ones(Float32, n_context))
            end

            # Compute other channels.
            for j in 1:num_channels
                channel = weights * y_context[:, j, i]
                if perform_encoding
                    channel ./= channels[1] .+ 1f-8
                end
                push!(channels, channel)
            end

            push!(batches, hcat(channels...))
        end
        ref = insert_dim(cat(batches..., dims=3), pos=2)

        # Check that the brute-force calculation lines up with the layer.
        @test out ≈ ref
    end
end
