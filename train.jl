using ArgParse
using Pkg

Pkg.instantiate()

# Parse command line arguments.
parser = ArgParseSettings()
@add_arg_table! parser begin
    "--data"
        help =
            "Data set: eq-small, eq, matern52, eq-mixture, noisy-mixture, " *
            "weakly-periodic, sawtooth, or mixture. " *
            "Append \"-noisy\" to a data set to make it noisy."
        arg_type = String
        required = true
    "--model"
        help =
            "Model: conv[c]np, corconvcnp, a[c]np, or [c]np. " *
            "Append \"-global-{mean,sum}\" to introduce a global latent variable. " *
            "Append \"-amortised-{mean,sum}\" to use amortised observation noise. " *
            "Append \"-het\" to use heterogeneous observation noise."
        arg_type = String
        required = true
    "--num-samples"
        help =
            "Number of samples to estimate the training loss. Defaults to 20 for " *
            "\"loglik\" and 5 for \"elbo\"."
        arg_type = Int
    "--batch-size"
        help = "Batch size."
        arg_type = Int
        default = 16
    "--loss"
        help = "Loss: loglik, loglik-iw, or elbo."
        arg_type = String
        required = true
    "--starting-epoch"
        help = "Set to a number greater than one to continue training."
        arg_type = Int
        default = 1
    "--epochs"
        help = "Number of epochs to training for."
        arg_type = Int
        default = 20
    "--evaluate"
        help = "Evaluate model."
        action = :store_true
    "--evaluate-iw"
        help = "Force to use importance weighting for the evaluation objective."
        action = :store_true
    "--evaluate-no-iw"
        help = "Force to NOT use importance weighting for the evaluation objective."
        action = :store_true
    "--evaluate-num-samples"
        help = "Number of samples to estimate the evaluation loss."
        arg_type = Int
        default = 4096
    "--evaluate-only-within"
        help = "Evaluate with only the task of interpolation within training range."
        action = :store_true
    "--models-dir"
        help = "Directory to store models in."
        arg_type = String
        default = "models"
    "--bson"
        help = "Directly specify the file to save the model to and load it from."
        arg_type = String
end
args = parse_args(parser)

using BSON
using NeuralProcesses
using NeuralProcesses.Experiment
using Distributions
using Flux
using Stheno
using Tracker

# Determine the noise level.
if endswith(args["data"], "-noisy")
    trimmed_data_name = args["data"][1:end - length("-noisy")]
    noise = 0.05^2  # Use a total error of 0.05.
else
    trimmed_data_name = args["data"]
    noise = 1e-8  # Use very little noise, but still some for regularisation.
end

# Set up experiment.
if trimmed_data_name == "eq-small"
    process = GP(stretch(EQ(), 1 / 0.25), GPC())
    receptive_field = 1f0
    points_per_unit = 32f0
    num_context = DiscreteUniform(0, 50)
    num_target = DiscreteUniform(50, 50)
    num_channels = 16
    dim_embedding = 32
    num_context_eval = DiscreteUniform(0, 10)
elseif trimmed_data_name == "eq"
    process = GP(stretch(EQ(), 1 / 0.25), GPC())
    receptive_field = 2f0
    points_per_unit = 64f0
    num_context = DiscreteUniform(0, 50)
    num_target = DiscreteUniform(50, 50)
    num_channels = 64
    dim_embedding = 128
    num_context_eval = DiscreteUniform(0, 10)
elseif trimmed_data_name == "matern52"
    process = GP(stretch(Matern52(), 1 / 0.25), GPC())
    receptive_field = 2f0
    points_per_unit = 64f0
    num_context = DiscreteUniform(0, 50)
    num_target = DiscreteUniform(50, 50)
    num_channels = 64
    dim_embedding = 128
    num_context_eval = DiscreteUniform(0, 10)
elseif trimmed_data_name == "noisy-mixture"
    process = GP(stretch(EQ(), 1 / 0.25) + EQ() + 1e-3 * Stheno.Noise(), GPC())
    receptive_field = 4f0
    points_per_unit = 64f0
    num_context = DiscreteUniform(0, 50)
    num_target = DiscreteUniform(50, 50)
    num_channels = 64
    dim_embedding = 128
    num_context_eval = DiscreteUniform(0, 10)
elseif trimmed_data_name == "eq-mixture"
    process = GP(stretch(EQ(), 1 / 0.25) + EQ(), GPC())
    receptive_field = 4f0
    points_per_unit = 64f0
    num_context = DiscreteUniform(0, 50)
    num_target = DiscreteUniform(50, 50)
    num_channels = 64
    dim_embedding = 128
    num_context_eval = DiscreteUniform(0, 10)
elseif trimmed_data_name == "weakly-periodic"
    process = GP(stretch(EQ(), 1 / 0.5) * stretch(Stheno.PerEQ(1), 1 / 0.25), GPC())
    receptive_field = 4f0
    points_per_unit = 64f0
    num_context = DiscreteUniform(0, 50)
    num_target = DiscreteUniform(50, 50)
    num_channels = 64
    dim_embedding = 128
    num_context_eval = DiscreteUniform(0, 10)
elseif trimmed_data_name == "sawtooth"
    process = Sawtooth()
    receptive_field = 16f0
    points_per_unit = 64f0
    num_context = DiscreteUniform(0, 100)
    num_target = DiscreteUniform(100, 100)
    num_channels = 64
    dim_embedding = 128
    num_context_eval = DiscreteUniform(0, 10)
elseif trimmed_data_name == "mixture"
    process = Mixture(
        GP(stretch(EQ(), 1 / 0.25), GPC()),
        GP(stretch(Matern52(), 1 / 0.25), GPC()),
        GP(stretch(EQ(), 1 / 0.25) + EQ() + 1e-3 * Stheno.Noise(), GPC()),
        GP(stretch(EQ(), 1 / 0.5) * stretch(Stheno.PerEQ(1), 1 / 0.25), GPC()),
        Sawtooth()
    )
    receptive_field = 16f0
    points_per_unit = 64f0
    num_context = DiscreteUniform(0, 100)
    num_target = DiscreteUniform(100, 100)
    num_channels = 64
    dim_embedding = 128
    num_context_eval = DiscreteUniform(0, 10)
else
    error("Unknown data \"" * args["data"] * "\".")
end

# Set the loss.
if args["model"] in ["convcnp", "corconvcnp", "acnp", "cnp"]
    # Determine training loss.
    if args["loss"] == "loglik"
        # Use a single sample: there is nothing random.
        loss(xs...) = NeuralProcesses.loglik(xs..., num_samples=1)
    elseif args["loss"] in ["elbo", "loglik-iw"]
        error("Losses \"elbo\" and \"loglik-iw\" not applicable conditional models.")
    else
        error("Unknown loss \"" * args["loss"] * "\".")
    end

    # Use the train loss for evaluation.
    eval_loss = loss
elseif args["model"] in [
    "convnp",
    "convnp-global-mean", "convnp-global-sum",
    "convnp-amortised-mean", "convnp-amortised-sum",
    "convnp-het",
    "anp",
    "anp-amortised-mean", "anp-amortised-sum",
    "anp-het",
    "np",
    "np-amortised-mean", "np-amortised-sum",
    "np-het"
]
    # Determine training loss.
    if args["loss"] == "loglik"
        if !isnothing(args["num-samples"])
            num_samples = args["num-samples"]
            args["loss"] *= "-$num_samples"  # Incorporate number of samples in the loss.
        else
            num_samples = 20
        end
        loss(xs...) = NeuralProcesses.loglik(
            xs...,
            num_samples=num_samples,
            importance_weighted=false,
            fixed_σ_epochs=3
        )
        eval_importance_weighted = false  # Encoder is not suited for IW.
    elseif args["loss"] == "loglik-iw"
        if !isnothing(args["num-samples"])
            num_samples = args["num-samples"]
            args["loss"] *= "-$num_samples"  # Incorporate number of samples in the loss.
        else
            num_samples = 20
        end
        loss(xs...) = NeuralProcesses.loglik(
            xs...,
            num_samples=num_samples,
            importance_weighted=true,
            fixed_σ_epochs=3
        )
        eval_importance_weighted = true  # Encoder is suited for IW!
    elseif args["loss"] == "elbo"
        if !isnothing(args["num-samples"])
            num_samples = args["num-samples"]
            args["loss"] *= "-$num_samples"  # Incorporate number of samples in the loss.
        else
            num_samples = 5
        end
        loss(xs...) = NeuralProcesses.elbo(
            xs...,
            num_samples=num_samples,
            fixed_σ_epochs=3
        )
        eval_importance_weighted = true  # Encoder is suited for IW!
    else
        error("Unknown loss \"" * args["loss"] * "\".")
    end

    # Check if `eval_importance_weighted` needs to be forced.
    if args["evaluate-iw"] && args["evaluate-no-iw"]
        error("Cannot set both \"--evaluate-iw\" and \"--evaluate-no-iw\".")
    elseif args["evaluate-iw"]
        println("Force using importance weighting for evaluation objective.")
        eval_importance_weighted = true
    elseif args["evaluate-no-iw"]
        println("Force NOT using importance weighting for evaluation objective.")
        eval_importance_weighted = false
    end

    # Use a high-sample log-EL for the eval loss.
    eval_loss(xs...) = NeuralProcesses.loglik(
        xs...,
        num_samples=args["evaluate-num-samples"],
        importance_weighted=eval_importance_weighted,
        fixed_σ_epochs=0
    )
else
    error("Unknown model \"" * args["model"] * "\".")
end

# Determine name of file to write model to.
if !isnothing(args["bson"])
    bson = args["bson"]
else
    bson =
        args["models-dir"] * "/" *
        args["model"] * "/" *
        args["loss"] * "/" *
        args["data"] * ".bson"
    mkpath(args["models-dir"] * "/" * args["model"] * "/" * args["loss"])
end

# Determine folder to output images.
path = "output/" * args["model"] * "/" * args["loss"] * "/" * args["data"]
mkpath("output/" * args["model"] * "/" * args["loss"] * "/" * args["data"])

function build_data_gen(; x_context, x_target, num_context, num_target, batch_size)
    return DataGenerator(
        process,
        batch_size=batch_size,
        x_context=x_context,
        x_target=x_target,
        num_context=num_context,
        num_target=num_target,
        σ²=noise
    )
end

if args["evaluate"]
    # Use the best model for evaluation.
    model = best_model(bson) |> gpu
    report_num_params(model)

    # Use a batch size of one to support a high number of samples. We alleviate the increase
    # in variance of the objective by using a high number of batches.
    batch_size = 1
    num_batches = 5000

    # Determine which evaluation tasks to perform.
    tasks = [(
        "interpolation on training range",
        build_data_gen(
            x_context=Uniform(-2, 2),
            x_target=Uniform(-2, 2),
            num_context=num_context_eval,
            num_target=num_target,
            batch_size=batch_size
        )
    )]
    if !args["evaluate-only-within"]
        push!(tasks, (
            "interpolation beyond training range",
            build_data_gen(
                x_context=Uniform(2, 6),
                x_target=Uniform(2, 6),
                num_context=num_context_eval,
                num_target=num_target,
                batch_size=batch_size
            )
        ))
        push!(tasks, (
            "extrapolation beyond training range",
            build_data_gen(
                x_context=Uniform(-2, 2),
                x_target=UniformUnion(Uniform(-4, -2), Uniform(2, 4)),
                num_context=num_context_eval,
                num_target=num_target,
                batch_size=batch_size
            )
        ))
    end

    # Perform evaluation tasks with `epoch` set to 1000.
    for (name, data_gen) in tasks
        println("Evaluation task: $name")
        eval_model(model, eval_loss, data_gen, 1000, num_batches=num_batches)
    end
else
    # Construct data generator for training.
    data_gen = build_data_gen(
        x_context=Uniform(-2, 2),
        x_target=Uniform(-2, 2),
        num_context=num_context,
        num_target=num_target,
        batch_size=args["batch-size"]
    )

    if args["starting-epoch"] > 1
        # Continue training from most recent model.
        model = recent_model(bson) |> gpu
    else
        # Instantiate a new model to start training. Ideally, the margin should be the
        # receptive field size, but that creates large memory requirements for models with
        # large receptive field.
        margin = 0f1
        if args["model"] == "convcnp"
            model = convcnp_1d(
                receptive_field=receptive_field,
                num_layers=8,
                num_channels=num_channels,
                points_per_unit=points_per_unit,
                margin=margin
            ) |> gpu
        elseif args["model"] == "corconvcnp"
            model = corconvcnp_1d(
                receptive_field_μ=receptive_field,
                receptive_field_Σ=min(receptive_field, 8f0),
                num_layers=8,
                num_channels=num_channels,
                points_per_unit_μ=points_per_unit,
                points_per_unit_Σ=20f0,
                margin=margin
            ) |> gpu
        elseif args["model"] in [
            "convnp",
            "convnp-global-mean", "convnp-global-sum",
            "convnp-amortised-mean", "convnp-amortised-sum",
            "convnp-het"
        ]
            if args["model"] == "convnp"
                num_global_channels = 0
                noise_type = "fixed"
                pooling_type = "mean"  # This doesn't matter, but must be set to something.
            elseif args["model"] == "convnp-global-mean"
                num_global_channels = 16
                noise_type = "fixed"
                pooling_type = "mean"
            elseif args["model"] == "convnp-global-sum"
                num_global_channels = 16
                noise_type = "fixed"
                pooling_type = "sum"
            elseif args["model"] == "convnp-amortised-mean"
                num_global_channels = 0
                noise_type = "amortised"
                pooling_type = "mean"
            elseif args["model"] == "convnp-amortised-sum"
                num_global_channels = 0
                noise_type = "amortised"
                pooling_type = "sum"
            elseif args["model"] == "convnp-het"
                num_global_channels = 0
                noise_type = "het"
                pooling_type = "mean"  # This doesn't matter, but must be set to something.
            else
                error("Unknown model \"" * args["model"] * "\".")
            end

            model = convnp_1d(
                receptive_field=receptive_field,
                num_encoder_layers=8,
                num_decoder_layers=8,
                num_encoder_channels=num_channels,
                num_decoder_channels=num_channels,
                num_latent_channels=16,
                num_global_channels=num_global_channels,
                points_per_unit=points_per_unit,
                margin=1f0,
                noise_type=noise_type,
                pooling_type=pooling_type,
                σ=5f-2,
                learn_σ=false
            ) |> gpu
        elseif args["model"] == "acnp"
            model = acnp_1d(
                dim_embedding=dim_embedding,
                num_encoder_heads=8,
                num_encoder_layers=6,
                num_decoder_layers=6
            ) |> gpu
        elseif args["model"] in [
            "anp",
            "anp-amortised-mean", "anp-amortised-sum",
            "anp-het"
        ]
            if args["model"] == "anp"
                noise_type = "fixed"
                pooling_type = "mean"  # This doesn't matter, but must be set to something.
            elseif args["model"] == "anp-amortised-mean"
                noise_type = "amortised"
                pooling_type = "mean"
            elseif args["model"] == "anp-amortised-sum"
                noise_type = "amortised"
                pooling_type = "sum"
            elseif args["model"] == "anp-het"
                noise_type = "het"
                pooling_type = "mean"  # This doesn't matter, but must be set to something.
            else
                error("Unknown model \"" * args["model"] * "\".")
            end

            model = anp_1d(
                dim_embedding=dim_embedding,
                num_encoder_heads=8,
                num_encoder_layers=6,
                num_decoder_layers=6,
                noise_type=noise_type,
                pooling_type=pooling_type,
                σ=5f-2,
                learn_σ=false
            ) |> gpu
        elseif args["model"] == "cnp"
            model = cnp_1d(
                dim_embedding=dim_embedding,
                num_encoder_layers=6,
                num_decoder_layers=6
            ) |> gpu
        elseif args["model"] in [
            "np",
            "np-amortised-mean", "np-amortised-sum",
            "np-het"
        ]
            if args["model"] == "np"
                noise_type = "fixed"
                pooling_type = "mean"  # This doesn't matter, but must be set to something.
            elseif args["model"] == "np-amortised-mean"
                noise_type = "amortised"
                pooling_type = "mean"
            elseif args["model"] == "np-amortised-sum"
                noise_type = "amortised"
                pooling_type = "sum"
            elseif args["model"] == "np-het"
                noise_type = "het"
                pooling_type = "mean"  # This doesn't matter, but must be set to something.
            else
                error("Unknown model \"" * args["model"] * "\".")
            end

            model = np_1d(
                dim_embedding=dim_embedding,
                num_encoder_layers=6,
                num_decoder_layers=6,
                noise_type=noise_type,
                pooling_type=pooling_type,
                σ=5f-2,
                learn_σ=false
            ) |> gpu
        else
            error("Unknown model \"" * args["model"] * "\".")
        end
    end

    report_num_params(model)

    train!(
        model,
        loss,
        data_gen,
        ADAM(5e-4),
        bson=bson,
        starting_epoch=args["starting-epoch"],
        tasks_per_epoch=2^14,
        epochs=args["epochs"],
        path=path
    )
end
