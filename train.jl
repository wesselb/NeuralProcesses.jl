using Pkg

Pkg.activate(".")
Pkg.resolve()
Pkg.instantiate()

using ArgParse
using BSON
using ConvCNPs
using ConvCNPs.Experiment
using Flux
using Flux.Tracker
using Stheno
using Distributions

# Parse command line arguments.
parser = ArgParseSettings()
@add_arg_table! parser begin
    "--data"
        help = "Data set: eq-small, eq, matern52, noisy-mixture, weakly-periodic, sawtooth, or mixture."
        arg_type = String
        required = true
    "--model"
        help = "Model: convcnp, convnp, convnp-global, anp, or np."
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
    "--evaluate"
        help = "Evaluate model."
        action = :store_true
    "--epochs"
        help = "Number of epochs to training for."
        arg_type = Int
        default = 20
end
args = parse_args(parser)

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
    num_channels = 32
    dim_embedding = 128
elseif args["data"] == "mixture"
    process = Mixture(GP(Stheno.ConstKernel(1.0), GPC()), GP(stretch(eq(), 1 / 0.25), GPC()))
    receptive_field = 2f0
    points_per_unit = 64f0
    num_context = DiscreteUniform(0, 50)
    num_target = DiscreteUniform(50, 50)
    num_channels = 64
    dim_embedding = 128
else
    error("Unknown data \"" * args["data"] * "\".")
end

# Determine name of file to write model to and folder to output images.
bson = "models/" * args["model"] * "/" * args["loss"] * "/" * args["data"] * ".bson"
path = "output/" * args["model"] * "/" * args["loss"] * "/" * args["data"]

# Ensure that the appropriate directories exist.
mkpath("models/" * args["model"] * "/" * args["loss"] * "/" * args["data"])
mkpath("output/" * args["model"] * "/" * args["loss"] * "/" * args["data"])

# Set the loss.
if args["model"] == "convcnp"
    if args["loss"] == "loglik"
        loss = ConvCNPs.loglik
    elseif args["loss"] in ["elbo", "loglik-iw"]
        error("Losses \"elbo\" and \"loglik-iw\" not applicable to the ConvCNP.")
    else
        error("Unknown loss \"" * args["loss"] * "\".")
    end
elseif args["model"] in ["convnp", "convnp-global", "anp", "np"]
    if args["loss"] == "loglik"
        loss(xs...) = ConvCNPs.loglik(xs..., num_samples=20, importance_weighted=false)
    elseif args["loss"] == "loglik-iw"
        loss(xs...) = ConvCNPs.loglik(xs..., num_samples=20, importance_weighted=true)
    elseif args["loss"] == "elbo"
        loss(xs...) = ConvCNPs.elbo(xs..., num_samples=5)
    else
        error("Unknown loss \"" * args["loss"] * "\".")
    end
else
    error("Unknown model \"" * args["model"] * "\".")
end

function build_data_gen(x_context, x_target)
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
    # Loop over various data generators for various tasks.
    for (name, data_gen) in [
        (
            "interpolation on training range",
            build_data_gen(Uniform(-2, 2), Uniform(-2, 2))
        ),
        (
            "interpolation beyond training range",
            build_data_gen(Uniform(-4, 4), UniformUnion(Uniform(-4, -2), Uniform(2, 4)))
        ),
        (
            "extrapolation beyond training range",
            build_data_gen(Uniform(-2, 2), UniformUnion(Uniform(-4, -2), Uniform(2, 4)))
        )
    ]
        println("Evaluation task: $name")
        model = best_model(bson) |> gpu  # Use the best model for evaluation.
        report_num_params(model)
        eval_model(model, loss, data_gen, 100, num_batches=10000)
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
        elseif args["model"] in ["convnp", "convnp-global"]
            model = convnp_1d(
                receptive_field=receptive_field,
                num_encoder_layers=8,
                num_decoder_layers=8,
                num_encoder_channels=div(num_channels, 2),
                num_decoder_channels=div(num_channels, 2),
                num_latent_channels=8,
                points_per_unit=points_per_unit,
                margin=1f0,
                σ=2f-2,
                learn_σ=false,
                global_variable=args["model"] == "convnp-global"
            ) |> gpu
        elseif args["model"] == "anp"
            model = anp_1d(
                dim_embedding=dim_embedding,
                num_encoder_heads=8,
                num_encoder_layers=3,
                num_decoder_layers=3,
                σ=2f-2,
                learn_σ=false
            ) |> gpu
        elseif args["model"] == "np"
            model = np_1d(
                dim_embedding=dim_embedding,
                num_encoder_layers=3,
                num_decoder_layers=3,
                σ=2f-2,
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
