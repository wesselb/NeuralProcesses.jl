using Pkg

Pkg.activate(".")
Pkg.resolve()
Pkg.instantiate()

using ConvCNPs
using Flux
using GPUArrays
using Random
using Stheno
using StatsBase
using Distributions
using Plots

# GPUArrays.allowscalar(false)


function plot_task(model, epoch)
    x = LinRange(-3, 3, 400)
    task = 1

    batch = gpu(data_gen(1)[1])
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

    mkpath("output")
    savefig(plt, "output/epoch$epoch.pdf")
end


function loss(model, x_context, y_context, x_target, y_target)
    return -mean(gaussian_logpdf(
        y_target,
        model(x_context, y_context, x_target)...
    ))
end


function data_to_gpu(data)
    return [(
        x_context=gpu(Array{Float32}(x.x_context)),
        y_context=gpu(Array{Float32}(x.y_context)),
        x_target=gpu(Array{Float32}(x.x_target)),
        y_target=gpu(Array{Float32}(x.y_target))
    ) for x in data]
end


function eval_model(model; num_batches=16)
    data = data_to_gpu(data_gen(num_batches))
    res = mean(map(x -> loss(model, x...), data))
    loss_value = Flux.data(res)
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

# Use the SimpleConv architecture.
conv = Chain(
    Conv((1,), 2=>8, pad=0),
    Conv((5,), 8=>16, pad=2, relu),
    Conv((5,), 16=>32, pad=2, relu),
    Conv((5,), 32=>16, pad=2, relu),
    Conv((5,), 16=>8, pad=2, relu),
    Conv((1,), 8=>2, pad=0),
)
arch = (conv=conv, points_per_unit=32, multiple=1)

# Use an architecture with depthwise separable convolutions.
# arch = build_conv(scale * 2, 4, 8)

# Instantiate ConvCNP model.
model = convcnp_1d(arch; margin = scale * 2)

model = model |> gpu

println(eval_model(model))

# Configure training.
opt = ADAM(5e-4)
EPOCHS = 100
TASKS_PER_EPOCH = 256

for epoch in 1:EPOCHS
    println("Epoch: $epoch")
    Flux.train!(
        (xs...) -> loss(model, xs...),
        Flux.params(model),
        data_to_gpu(data_gen(TASKS_PER_EPOCH)),
        opt,
        cb = Flux.throttle(() -> eval_model(model), 10)
    )
    eval_model(model; num_batches=128)
    plot_task(model, epoch)
end
