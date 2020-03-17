using Pkg

Pkg.activate(".")
Pkg.resolve()
Pkg.instantiate()

using Flux
using ConvCNPs
using Random
using Stheno
using StatsBase
using Distributions
using Plots


function plot_task(model, epoch)
    x = LinRange(-3, 3, 400)
    task = 1

    batch = data_gen(1)[1]
    y_mean, y_std = model(batch.x_context, batch.y_context, repeat(x, 1, 1, 16))
    y_mean = y_mean[:, 1, task].data
    y_std = y_std[:, 1, task].data

    plt = plot()

    # Scatter context set
    scatter!(
        plt,
        batch.x_context[:, 1, task],
        batch.y_context[:, 1, task],
        c=:black,
        label="Context set"
    )
    scatter!(
        plt,
        batch.x_target[:, 1, task],
        batch.y_target[:, 1, task],
        c=:red,
        label="Target set"
    )

    # Plot prediction
    plot!(plt, x, y_mean, c=:green, label="Model Output")
    plot!(
        plt,
        x,
        [y_mean y_mean],
        fillrange=[y_mean .+ 2y_std y_mean .- y_std],
        fillalpha=0.2,
        c=:green,
        label=""
    )

    savefig(plt, "output/epoch$epoch.png")
end


function loss(model, x_context, y_context, x_target, y_target)
    return -mean(gaussian_logpdf(
        y_target,
        model(x_context, y_context, x_target)...
    ))
end


function eval_model(model; num_batches=16)
    loss_value = Flux.data(mean(map(x -> loss(model, x...), data_gen(num_batches))))
    println("Test loss: $loss_value ($num_batches batches)")
end


# Construct data generator.
scale = 0.25
k = stretch(matern52(), 1 / scale)
data_gen = DataGenerator(
    k;
    batch_size=16,
    x_dist=Uniform(-2, 2),
    max_context_points=50,
    max_target_points=50
)

# # Use the SimpleConv architecture.
# conv = Chain(
#     Conv((1,), 2=>8, pad=0),
#     Conv((5,), 8=>16, pad=2, relu),
#     Conv((5,), 16=>32, pad=2, relu),
#     Conv((5,), 32=>16, pad=2, relu),
#     Conv((5,), 16=>8, pad=2, relu),
#     Conv((1,), 8=>2, pad=0),
# )
# arch = (conv=conv, points_per_unit=32, multiple=1)

# Use an architecture with depthwise separable convolutions.
arch = build_conv(scale * 2, 4, 8)

# Instantiate ConvCNP model.
model = convcnp_1d(arch; margin = scale * 2)

# Configure training.
opt = ADAM(5e-4)
EPOCHS = 100
TASKS_PER_EPOCH = 256

for epoch in 1:EPOCHS
    println("Epoch: $epoch")
    Flux.train!(
        (xs...) -> loss(model, xs...),
        Flux.params(model),
        data_gen(TASKS_PER_EPOCH),
        opt,
        cb = Flux.throttle(() -> eval_model(model), 10)
    )
    eval_model(model; num_batches=128)
    plot_task(model, epoch)
end
