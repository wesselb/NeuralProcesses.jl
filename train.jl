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
        help = "Data set."
        arg_type = String
        required = true
    "--model"
        help = "Model."
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
    num_context = DiscreteUniform(3, 50)
    num_target = DiscreteUniform(3, 50)
    receptive_field = 1f0
    points_per_unit = 32f0
    num_channels = 16
elseif args["data"] == "eq"
    process = GP(stretch(eq(), 1 / 0.25), GPC())
    num_context = DiscreteUniform(3, 50)
    num_target = DiscreteUniform(3, 50)
    receptive_field = 2f0
    points_per_unit = 64f0
    num_channels = 64
elseif args["data"] == "matern52"
    process = GP(stretch(matern52(), 1 / 0.25), GPC())
    num_context = DiscreteUniform(3, 50)
    num_target = DiscreteUniform(3, 50)
    receptive_field = 2f0
    points_per_unit = 64f0
    num_channels = 64
elseif args["data"] == "weakly-periodic"
    process = GP(stretch(eq(), 1 / 0.5) * stretch(Stheno.PerEQ(), 1 / 0.25), GPC())
    num_context = DiscreteUniform(3, 50)
    num_target = DiscreteUniform(3, 50)
    receptive_field = 4f0
    points_per_unit = 64f0
    num_channels = 64
elseif args["data"] == "sawtooth"
    process = Sawtooth()
    num_context = DiscreteUniform(3, 100)
    num_target = DiscreteUniform(3, 100)
    receptive_field = 16f0
    points_per_unit = 64f0
    num_channels = 32
else
    error("Unknown data \"" * args["data"] * "\".")
end

# Determine name of file to write model to and folder to output images.
bson = "models/" * args["model"] * "/" * args["data"] * ".bson"
path = "output/" * args["model"] * "/" * args["data"]

# Ensure that the directories exist.
mkpath("models/" * args["model"])
mkpath(path)

# Construct data generator.
data_gen = DataGenerator(
    process;
    batch_size=16,
    x=Uniform(-2, 2),
    num_context=num_context,
    num_target=num_target
)

# Set the loss.
if args["model"] == "convcnp"
    loss = ConvCNPs.loss
elseif args["model"] == "convnp"
    loss(xs...) = ConvCNPs.loss(xs..., num_samples=5)
else
    error("Unknown model \"" * args["model"] * "\".")
end

function report_num_params(model)
    println("Number of parameters: ", sum(map(length, Flux.params(model))))
end

if args["evaluate"]
    # Use the best models for evaluation.
    for checkpoint in load_checkpoints(bson).top
        model = checkpoint.model |> gpu
        report_num_params(model)
        eval_model(model, loss, data_gen, 100, num_batches=10000)
    end
else
    if args["starting-epoch"] > 1
        # Continue training from most recent model.
        model = recent_model(bson) |> gpu
    else
        # Instantiate a new model to start training.
        if args["model"] == "convcnp"
            model = convcnp_1d(
                receptive_field=receptive_field,
                num_layers=8,
                num_channels=num_channels,
                points_per_unit=points_per_unit,
                margin=receptive_field
            ) |> gpu
        elseif args["model"] == "convnp"
            model = convnp_1d(
                receptive_field=receptive_field,
                encoder_layers=8,
                decoder_layers=2,
                encoder_channels=num_channels,
                decoder_channels=num_channels,
                latent_channels=2,
                points_per_unit=points_per_unit,
                margin=receptive_field
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
        batches_per_epoch=2048,
        starting_epoch=args["starting-epoch"],
        epochs=args["epochs"],
        path=path
    )
end
