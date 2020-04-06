using Pkg

Pkg.activate(".")
Pkg.resolve()
Pkg.instantiate()

using BSON: @save
using ConvCNPs
using Flux
using Random
using Stheno
using StatsBase
using Distributions
using Plots
using Printf
using GPUArrays

GPUArrays.allowscalar(false)
pyplot()

function plot_task(model, epoch, plot_true = (plt, x_context, y_context, x) -> nothing)
    x = gpu(collect(range(-3, 3, length=400)))

    # Extract the first task.
    x_context, y_context, x_target, y_target = map(x -> x[:, 1, 1], data_gen(1)[1])

    # Run model. Take care of the dimensionality of all objects and bringing
    # them to the GPU and back.
    expand(x) = gpu(reshape(x, length(x), 1, 1))
    y_mean, y_var = map(
        x -> Flux.data(cpu(x[:, 1, 1])),
        model(expand.((x_context, y_context, x))...)
    )
    x = cpu(x)

    plt = plot()

    # Scatter context set.
    scatter!(plt, x_context, y_context, c=:black, label="Context set", dpi=200)
    scatter!(plt, x_target, y_target, c=:red, label="Target set", dpi=200)

    # Plot prediction of true, underlying model.
    plot_true(plt, x_context, y_context, x)

    # Plot prediction.
    plot!(plt, x, y_mean, c=:green, label="Model output", dpi=200)
    plot!(plt, x, [y_mean y_mean],
        fillrange=[y_mean .+ 2 .* sqrt.(y_var) y_mean .- 2 .* sqrt.(y_var)],
        fillalpha=0.2,
        c=:green,
        label="",
        dpi=200
    )

    mkpath("output")
    savefig(plt, "output/epoch$epoch.png")
end

function make_plot_true(process)
    return (plt, x_context, y_context, x) -> nothing
end

function make_plot_true(process::GP)
    function plot_true(plt, x_context, y_context, x)
        x_context, y_context, x = map(z -> Float64.(z), (x_context, y_context, x))
        posterior = process | Obs(process(x_context, 1e-10) ← y_context)
        margs = marginals(posterior(x))
        plot!(plt, x, mean.(margs); c=:blue, label="GP", dpi=200)
        plot!(
            plt,
            x,
            mean.(margs) .- 2 .* std.(margs);
            c=:blue,
            linestyle=:dash,
            label="",
            dpi=200
        )
        plot!(
            plt,
            x,
            mean.(margs) .+ 2 .* std.(margs);
            c=:blue,
            linestyle=:dash,
            label="",
            dpi=200
        )
    end
    return plot_true
end

function loss(model, x_context, y_context, x_target, y_target)
    return -mean(gaussian_logpdf(y_target, model(x_context, y_context, x_target)...))
end

function eval_model(model; num_batches=16)
    loss_value = Flux.data(mean(map(x -> loss(model, gpu.(x)...), data_gen(num_batches))))
    @printf("Test loss: %.3f (%d batches)\n", loss_value, num_batches)
end

function train!(model, data_gen, opt, epochs=100, batches_per_epoch=2048)
    # Evaluate once before training.
    eval_model(model)

    for epoch in 1:epochs
        println("Epoch: $epoch")
        Flux.train!(
            (xs...) -> loss(model, gpu.(xs)...),
            Flux.params(model),
            data_gen(batches_per_epoch),
            opt,
            cb = Flux.throttle(() -> eval_model(model), 20)
        )

        eval_model(model; num_batches=128)
        plot_task(model, epoch, make_plot_true(data_gen.process))

        model_cpu = model |> cpu
        @save "matern52.bson" model_cpu
    end

    return model
end

# Construct data generator. The model's effective predictive extent is the scale.
scale = 0.25f0
process = GP(stretch(matern52(), 1 / Float64(scale)), GPC())
data_gen = DataGenerator(
    process;
    batch_size=8,
    x_dist=Uniform(-2, 2),
    max_context_points=10,
    num_target_points=50
)

# Use an architecture with depthwise separable convolutions.
arch = build_conv_1d(6scale, 6, 16; points_per_unit=30f0)

# Instantiate ConvCNP model.
model = convcnp_1d(arch; margin=4scale) |> gpu

# Configure training.
opt = ADAM(1e-3)

model = train!(model, data_gen, opt)
