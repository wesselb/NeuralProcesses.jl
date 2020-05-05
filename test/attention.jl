@testset "attention.jl" begin
    @testset "Attention" begin
        dim_x = 2
        dim_y = 3
        dim_embedding = 4
        num_channels = 5
        batch_size = 6
        n = 7
        m = 8

        layer = ConvCNPs.untrack(attention(
            dim_x=dim_x,
            dim_y=dim_y,
            dim_embedding=dim_embedding,
            num_channels=num_channels
        ))

        x_context = randn(Float32, n, dim_x, batch_size)
        y_context = randn(Float32, n, dim_y, batch_size)
        x_target = randn(Float32, m, dim_x, batch_size)

        # Perform encodings.
        keys = layer.encoder_x(x_context)
        queries = layer.encoder_x(x_target)
        values = layer.encoder_xy(cat(x_context, y_context, dims=2))

        # Brute-force the attention computation.
        embeddings = zeros(Float32, m, dim_embedding, num_channels, batch_size)
        for c = 1:num_channels
            for b = 1:batch_size
                # Calculate weights.
                weights = Array{Float32}(undef, n, m)
                for i = 1:n, j = 1:m
                    weights[i, j] = exp(dot(keys[i, :, c, b], queries[j, :, c, b]))
                end
                for j = 1:m
                    weights[:, j] ./= sum(weights[:, j])
                end

                # Calculate embeddings.
                for i = 1:n, j = 1:m
                    embeddings[j, :, c, b] .+= values[i, :, c, b] .* weights[i, j]
                end
            end
        end
        reference = layer.mixer(embeddings)

        # Check that the layer lines up with the brute-force reference.
        @test layer(x_context, y_context, x_target) ≈ reference
    end

    @testset "BatchedLinear" begin
        layer = ConvCNPs.untrack(BatchedLinear(2, 3))
        x = randn(10, 2, 4, 5)
        y = Array{Float32}(undef, 10, 3, 4, 5)
        for i = 1:4, j = 1:5
            y[:, :, i, j] = x[:, :, i, j] * layer.w .+ layer.b
        end
        @test layer(x) ≈ y
    end
end
