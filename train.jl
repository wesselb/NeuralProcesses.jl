using Pkg

Pkg.activate(".")
Pkg.resolve()
Pkg.instantiate()

using ConvCNPs
using Flux
using Random
using Stheno
using StatsBase
using Distributions
using Plots

function plot_task(model, epoch)
    # Extract the first task.
    x_context, y_context, x_target, y_target = map(x -> cpu(x[:, 1, 1]), data_gen(1)[1])

    # Run model. Take care of the dimensionality of all objects.
    expand(x) = reshape(x, length(x), 1, 1)
    y_mean, y_std = map(
        x -> Flux.data(cpu(x[:, 1, 1])),
        model(expand.((x_context, y_context, x))...)
    )

    plt = plot()

    # Scatter context set
    scatter!(plt, x_context, y_context, c=:black, label="Context set")
    scatter!(plt, x_target, y_target, c=:red, label="Target set")

    # Plot prediction
    plot!(plt, x, y_mean, c=:green, label="Model Output")
    plot!(plt, x, [y_mean y_mean],
        fillrange=[y_mean .+ 2y_std y_mean .- 2y_std],
        fillalpha=0.2,
        c=:green,
        label=""
    )

    mkpath("output")
    savefig(plt, "output/epoch$epoch.pdf")
end

function loss(model, x_context, y_context, x_target, y_target)
    return -mean(gaussian_logpdf(y_target, model(x_context, y_context, x_target)...))
end

function eval_model(model; num_batches=16)
    loss_value = Flux.data(mean(map(x -> loss(model, x...), data_gen(num_batches))))
    println("Test loss: $loss_value ($num_batches batches)")
end

# Construct data generator.
scale = 0.25f0
k = stretch(matern52(), 1 / Float32(scale))  # Use `Float64`s for the data generation.
data_gen = DataGenerator(
    k;
    batch_size=16,
    x_dist=Uniform(-2, 2),
    max_context_points=50,
    max_target_points=50
)

# Use the SimpleConv architecture.
conv = Chain(
    Conv((1, 1), 2=>8, pad=0, sigmoid; init=Flux.glorot_normal),
    Conv((5, 1), 8=>16, pad=(2, 0), relu; init=Flux.glorot_normal),
    Conv((5, 1), 16=>32, pad=(2, 0), relu; init=Flux.glorot_normal),
    Conv((5, 1), 32=>16, pad=(2, 0), relu; init=Flux.glorot_normal),
    Conv((5, 1), 16=>8, pad=(2, 0), relu; init=Flux.glorot_normal),
    Conv((1, 1), 8=>2, pad=0; init=Flux.glorot_normal),
)
arch = (conv=conv, points_per_unit=32f0, multiple=1)

# Use an architecture with depthwise separable convolutions.
# arch = build_conv(scale * 2, 4, 8)

# Instantiate ConvCNP model.
model = convcnp_1d(arch; margin = scale * 2) |> gpu

# Evaluate once before training.
eval_model(model)

# Configure training.
opt = ADAM(3e-4)
EPOCHS = 100
TASKS_PER_EPOCH = 512

for epoch in 1:EPOCHS
    println("Epoch: $epoch")
    Flux.train!(
        (xs...) -> loss(model, xs...),
        Flux.params(model),
        data_gen(TASKS_PER_EPOCH),
        opt,
        cb = Flux.throttle(() -> eval_model(model), 10)
    )
    println("Epoch done")
    eval_model(model; num_batches=128)
    # plot_task(model, epoch)
end
