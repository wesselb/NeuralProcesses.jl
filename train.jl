using ArgParse
using Pkg

Pkg.instantiate()

# Parse command line arguments.
parser = ArgParseSettings()
@add_arg_table! parser begin
    "--data"
        help =
            "Data set: eq-small, eq, matern52, noisy-mixture, weakly-periodic, sawtooth, " *
            "or mixture."
        arg_type = String
        required = true
    "--model"
        help =
            "Model: convcnp, convnp[-{global,amortised}-{sum,mean}], " *
            "anp[-amortised-{sum,mean}], or np[-amortised-{sum,mean}]."
        arg_type = String
        required = true
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
    "--models-dir"
        help = "Directory to store models in."
        arg_type = String
        default = "models"
end
args = parse_args(parser)

using BSON
using ConvCNPs
using ConvCNPs.Experiment
using Distributions
using Flux
using Flux.Tracker
using Stheno

# Set up experiment.
if args["data"] == "eq-small"
    process = GP(stretch(eq(), 1 / 0.25), GPC())
    receptive_field = 1f0
    points_per_unit = 32f0
    num_context = DiscreteUniform(0, 50)
    num_target = DiscreteUniform(50, 50)
    num_channels = 16
    dim_embedding = 32
elseif args["data"] == "eq"
    process = GP(stretch(eq(), 1 / 0.25), GPC())
    receptive_field = 2f0
    points_per_unit = 64f0
    num_context = DiscreteUniform(0, 50)
    num_target = DiscreteUniform(50, 50)
    num_channels = 64
    dim_embedding = 128
elseif args["data"] == "matern52"
    process = GP(stretch(matern52(), 1 / 0.25), GPC())
    receptive_field = 2f0
    points_per_unit = 64f0
    num_context = DiscreteUniform(0, 50)
    num_target = DiscreteUniform(50, 50)
    num_channels = 64
    dim_embedding = 128
elseif args["data"] == "noisy-mixture"
    process = GP(stretch(eq(), 1 / 0.25) + eq() + 1e-3 * Stheno.Noise(), GPC())
    receptive_field = 4f0
    points_per_unit = 64f0
    num_context = DiscreteUniform(0, 50)
    num_target = DiscreteUniform(50, 50)
    num_channels = 64
    dim_embedding = 128
elseif args["data"] == "weakly-periodic"
    process = GP(stretch(eq(), 1 / 0.5) * stretch(Stheno.PerEQ(), 1 / 0.25), GPC())
    receptive_field = 4f0
    points_per_unit = 64f0
    num_context = DiscreteUniform(0, 50)
    num_target = DiscreteUniform(50, 50)
    num_channels = 64
    dim_embedding = 128
elseif args["data"] == "sawtooth"
    process = Sawtooth()
    receptive_field = 16f0
    points_per_unit = 64f0
    num_context = DiscreteUniform(0, 100)
    num_target = DiscreteUniform(100, 100)
    num_channels = 64
    dim_embedding = 128
elseif args["data"] == "mixture"
    process = Mixture(
        GP(stretch(eq(), 1 / 0.25), GPC()),
        GP(stretch(matern52(), 1 / 0.25), GPC()),
        GP(stretch(eq(), 1 / 0.25) + eq() + 1e-3 * Stheno.Noise(), GPC()),
        GP(stretch(eq(), 1 / 0.5) * stretch(Stheno.PerEQ(), 1 / 0.25), GPC()),
        Sawtooth()
    )
    receptive_field = 16f0
    points_per_unit = 64f0
    num_context = DiscreteUniform(0, 100)
    num_target = DiscreteUniform(100, 100)
    num_channels = 64
    dim_embedding = 128
else
    error("Unknown data \"" * args["data"] * "\".")
end

# Determine name of file to write model to and folder to output images.
bson =
    args["models-dir"] * "/" *
    args["model"] * "/" *
    args["loss"] * "/" *
    args["data"] * ".bson"
path = "output/" * args["model"] * "/" * args["loss"] * "/" * args["data"]

# Ensure that the appropriate directories exist.
mkpath(args["models-dir"] * "/" * args["model"] * "/" * args["loss"])
mkpath("output/" * args["model"] * "/" * args["loss"] * "/" * args["data"])

# Set the loss.
if args["model"] == "convcnp"
    # Determine training loss.
    if args["loss"] == "loglik"
        loss = ConvCNPs.loglik
    elseif args["loss"] in ["elbo", "loglik-iw"]
        error("Losses \"elbo\" and \"loglik-iw\" not applicable to the ConvCNP.")
    else
        error("Unknown loss \"" * args["loss"] * "\".")
    end

    # Use the train loss for evaluation.
    eval_loss = ConvCNPs.loglik
elseif args["model"] in [
    "convnp",
    "convnp-global-sum",
    "convnp-global-mean",
    "convnp-amortised-sum",
    "convnp-amortised-mean",
    "anp",
    "anp-amortised-sum",
    "anp-amortised-mean",
    "np",
    "np-amortised-sum",
    "np-amortised-mean"
]
    # Determine training loss.
    if args["loss"] == "loglik"
        loss(xs...) = ConvCNPs.loglik(xs..., num_samples=20, importance_weighted=false)
    elseif args["loss"] == "loglik-iw"
        loss(xs...) = ConvCNPs.loglik(xs..., num_samples=20, importance_weighted=true)
    elseif args["loss"] == "elbo"
        loss(xs...) = ConvCNPs.elbo(xs..., num_samples=5)
    else
        error("Unknown loss \"" * args["loss"] * "\".")
    end

    # Use a high-sample log-EL for the eval loss.
    eval_loss(xs...) = ConvCNPs.loglik(xs..., num_samples=100, importance_weighted=false)
else
    error("Unknown model \"" * args["model"] * "\".")
end

function build_data_gen(; x_context, x_target, num_context, num_target)
    return DataGenerator(
        process,
        batch_size=16,
        x_context=x_context,
        x_target=x_target,
        num_context=num_context,
        num_target=num_target
    )
end

if args["evaluate"]
    # Use the best model for evaluation.
    model = best_model(bson) |> gpu
    report_num_params(model)

    # Loop over various data generators for various tasks.
    for (name, data_gen) in [
        (
            "interpolation on training range",
            build_data_gen(
                x_context=Uniform(-2, 2),
                x_target=Uniform(-2, 2),
                num_context=num_context,
                num_target=num_target
            )
        ),
        (
            "interpolation beyond training range",
            build_data_gen(
                x_context=Uniform(2, 6),
                x_target=Uniform(2, 6),
                num_context=num_context,
                num_target=num_target
            )
        ),
        (
            "extrapolation beyond training range",
            build_data_gen(
                x_context=Uniform(-2, 2),
                x_target=UniformUnion(Uniform(-4, -2), Uniform(2, 4)),
                num_context=num_context,
                num_target=num_target
            )
        )
    ]
        println("Evaluation task: $name")
        eval_model(model, eval_loss, data_gen, 100, num_batches=5000)
    end
else
    # Construct data generator for training.
    data_gen = build_data_gen(Uniform(-2, 2), Uniform(-2, 2))

    if args["starting-epoch"] > 1
        # Continue training from most recent model.
        model = recent_model(bson) |> gpu
    else
        # Instantiate a new model to start training. Ideally, the margin should be the
        # receptive field size, but that creates large memory requirements for models with
        # large receptive field.
        if args["model"] == "convcnp"
            model = convcnp_1d(
                receptive_field=receptive_field,
                num_layers=8,
                num_channels=num_channels,
                points_per_unit=points_per_unit,
                margin=1f0
            ) |> gpu
        elseif args["model"] in [
            "convnp",
            "convnp-global-sum",
            "convnp-global-mean",
            "convnp-amortised-sum",
            "convnp-amortised-mean"
        ]
            if args["model"] == "convnp"
                num_global_channels = 0
                num_σ_channels = 0
                pooling_type = "sum"  # This doesn't matter, but must be set to something.
            elseif args["model"] == "convnp-global-sum"
                num_global_channels = 16
                num_σ_channels = 0
                pooling_type = "sum"
            elseif args["model"] == "convnp-global-mean"
                num_global_channels = 16
                num_σ_channels = 0
                pooling_type = "mean"
            elseif args["model"] == "convnp-amortised-sum"
                num_global_channels = 0
                num_σ_channels = 8
                pooling_type = "sum"
            elseif args["model"] == "convnp-amortised-mean"
                num_global_channels = 0
                num_σ_channels = 8
                pooling_type = "mean"
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
                num_σ_channels=num_σ_channels,
                points_per_unit=points_per_unit,
                margin=1f0,
                σ=2f-2,
                learn_σ=false,
                pooling_type=pooling_type
            ) |> gpu
        elseif args["model"] in ["anp", "anp-amortised-sum", "anp-amortised-mean"]
            if args["model"] == "anp"
                num_σ_channels = 0
                pooling_type = "sum"  # This doesn't matter, but must be set to something.
            elseif args["model"] == "anp-amortised-sum"
                num_σ_channels = 8
                pooling_type = "sum"
            elseif args["model"] == "anp-amortised-mean"
                num_σ_channels = 8
                pooling_type = "mean"
            else
                error("Unknown model \"" * args["model"] * "\".")
            end

            model = anp_1d(
                dim_embedding=dim_embedding,
                num_encoder_heads=8,
                num_encoder_layers=3,
                num_decoder_layers=3,
                num_σ_channels=num_σ_channels,
                σ=2f-2,
                learn_σ=false,
                pooling_type=pooling_type
            ) |> gpu
        elseif args["model"] in ["np", "np-amortised-sum", "np-amortised-mean"]
            if args["model"] == "np"
                num_σ_channels = 0
                pooling_type = "sum"  # This doesn't matter, but must be set to something.
            elseif args["model"] == "np-amortised-sum"
                num_σ_channels = 8
                pooling_type = "sum"
            elseif args["model"] == "np-amortised-mean"
                num_σ_channels = 8
                pooling_type = "mean"
            else
                error("Unknown model \"" * args["model"] * "\".")
            end

            model = np_1d(
                dim_embedding=dim_embedding,
                num_encoder_layers=3,
                num_decoder_layers=3,
                num_σ_channels=num_σ_channels,
                σ=2f-2,
                learn_σ=false,
                pooling_type=pooling_type
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
