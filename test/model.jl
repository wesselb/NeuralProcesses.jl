@testset "model" begin
    data_gen = DataGenerator(
        Sawtooth(),
        batch_size=4,
        x_context=Uniform(-2, 2),
        x_target=Uniform(-2, 2),
        num_context=DiscreteUniform(0, 3),
        num_target=DiscreteUniform(1, 3),
        σ²=1e-4
    )

    # Construct CNPs:
    cnps = [
        cnp_1d(
            dim_embedding=10,
            num_encoder_layers=2,
            num_decoder_layers=2
        ),
        acnp_1d(
            dim_embedding=10,
            num_encoder_layers=2,
            num_encoder_heads=3,
            num_decoder_layers=2
        ),
        convcnp_1d(
            receptive_field=1f0,
            num_layers=2,
            num_channels=2,
            points_per_unit=5f0
        )
    ]

    # Construct NPs:
    nps = Any[]
    for noise_type in ["fixed", "amortised", "het"], pooling_type in ["mean", "sum"]
        push!(nps, np_1d(
            dim_embedding=10,
            num_encoder_layers=2,
            num_decoder_layers=2,
            noise_type=noise_type,
            pooling_type=pooling_type
        ))
        push!(nps, anp_1d(
            dim_embedding=10,
            num_encoder_layers=2,
            num_encoder_heads=3,
            num_decoder_layers=2,
            noise_type=noise_type,
            pooling_type=pooling_type
        ))
        push!(nps, convnp_1d(
            receptive_field=1f0,
            num_encoder_layers=2,
            num_decoder_layers=3,
            num_encoder_channels=2,
            num_decoder_channels=1,
            num_latent_channels=3,
            num_global_channels=0,
            points_per_unit=5f0,
            noise_type=noise_type,
            pooling_type=pooling_type
        ))
        push!(nps, convnp_1d(
            receptive_field=1f0,
            num_encoder_layers=2,
            num_decoder_layers=3,
            num_encoder_channels=2,
            num_decoder_channels=1,
            num_latent_channels=3,
            num_global_channels=5,
            points_per_unit=5f0,
            noise_type=noise_type,
            pooling_type=pooling_type
        ))
    end

    # Construct all pairs of models and losses. CNPs:
    model_losses = Any[]
    for model in cnps
        push!(
            model_losses,
            (model, (xs...) -> NeuralProcesses.loglik(xs..., num_samples=1))
        )
    end

    # NPs:
    for model in nps
        push!(model_losses, (
            model,
            (xs...) -> NeuralProcesses.loglik(
                xs...,
                num_samples=2,
                batch_size=1,
                importance_weighted=false,
                fixed_σ_epochs=1
            )
        ))
        push!(model_losses, (
            model,
            (xs...) -> NeuralProcesses.loglik(
                xs...,
                num_samples=2,
                batch_size=1,
                importance_weighted=true,
                fixed_σ_epochs=1
            )
        ))
        push!(model_losses, (
            model,
            (xs...) -> NeuralProcesses.elbo(
                xs...,
                num_samples=2,
                fixed_σ_epochs=1
            )
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
