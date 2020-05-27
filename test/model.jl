@testset "model" begin
    data_gen = DataGenerator(
        Sawtooth(),
        batch_size=4,
        x_context=Uniform(-2, 2),
        x_target=Uniform(-2, 2),
        num_context=DiscreteUniform(0, 3),
        num_target=DiscreteUniform(1, 3)
    )

    # Construct models.
    convcnp = convcnp_1d(
        receptive_field=1f0,
        num_layers=2,
        num_channels=2,
        points_per_unit=5f0
    )
    convnp = convnp_1d(
        receptive_field=1f0,
        num_encoder_layers=2,
        num_decoder_layers=3,
        num_encoder_channels=2,
        num_decoder_channels=1,
        num_latent_channels=2,
        num_global_channels=0,
        num_σ_channels=0,
        points_per_unit=5f0
    )
    convnp_global_sum = convnp_1d(
        receptive_field=1f0,
        num_encoder_layers=2,
        num_decoder_layers=3,
        num_encoder_channels=2,
        num_decoder_channels=1,
        num_latent_channels=2,
        num_global_channels=2,
        num_σ_channels=0,
        points_per_unit=5f0,
        pooling_type="sum"
    )
    convnp_global_mean = convnp_1d(
        receptive_field=1f0,
        num_encoder_layers=2,
        num_decoder_layers=3,
        num_encoder_channels=2,
        num_decoder_channels=1,
        num_latent_channels=2,
        num_global_channels=2,
        num_σ_channels=0,
        points_per_unit=5f0,
        pooling_type="mean"
    )
    convnp_amortised_sum = convnp_1d(
        receptive_field=1f0,
        num_encoder_layers=2,
        num_decoder_layers=3,
        num_encoder_channels=2,
        num_decoder_channels=1,
        num_latent_channels=2,
        num_global_channels=0,
        num_σ_channels=8,
        points_per_unit=5f0,
        pooling_type="sum"
    )
    convnp_amortised_mean = convnp_1d(
        receptive_field=1f0,
        num_encoder_layers=2,
        num_decoder_layers=3,
        num_encoder_channels=2,
        num_decoder_channels=1,
        num_latent_channels=2,
        num_global_channels=0,
        num_σ_channels=8,
        points_per_unit=5f0,
        pooling_type="mean"
    )

    anp = anp_1d(
        dim_embedding=10,
        num_encoder_layers=2,
        num_encoder_heads=3,
        num_decoder_layers=2
    )
    np = np_1d(
        dim_embedding=10,
        num_encoder_layers=2,
        num_decoder_layers=2
    )

    # CNPs:
    model_losses = Any[(convcnp, ConvCNPs.loglik)]

    # NPs:
    for model in [
        convnp,
        convnp_global_sum,
        convnp_global_mean,
        convnp_amortised_sum,
        convnp_amortised_mean,
        anp,
        np
    ]
        push!(model_losses, (
            model,
            (xs...) -> ConvCNPs.loglik(xs..., num_samples=2, importance_weighted=false)
        ))
        push!(model_losses, (
            model,
            (xs...) -> ConvCNPs.loglik(xs..., num_samples=2, importance_weighted=true)
        ))
        push!(model_losses, (
            model,
            (xs...) -> ConvCNPs.elbo(xs..., num_samples=2)
        ))
    end

    for (model, loss) in model_losses
        # Test model training for a few epochs. The statements just have to not error.
        @test (report_num_params(model); true)
        @test (train!(
            model,
            loss,
            data_gen,
            ADAM(1e-4),
            bson=nothing,
            starting_epoch=1,
            epochs=2,
            tasks_per_epoch=200,
            path=nothing
        ); true)
    end
end
