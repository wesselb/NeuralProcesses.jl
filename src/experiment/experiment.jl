module Experiment

export predict, loss, eval_model, train!, report_num_params, plot_task

using ..ConvCNPs

using BSON
using Flux
using Flux.Tracker
using GPUArrays
using Plots
using Printf
import StatsBase: std
using Stheno

include("checkpoint.jl")

pyplot()

function eval_model(model, loss, data_gen, epoch; num_batches=256)
    model = ConvCNPs.untrack(model)
    values, target_sizes = map(
        x -> (loss(model, epoch, gpu.(x)...), size(x[3], 1)),
        data_gen(num_batches)
    )

    # Compute and print loss.
    loss_value, loss_error = _mean_error(values)
    @printf("Loss: %8.3f +- %7.3f (%d batches)\n", loss_value, loss_error, num_batches)

    # Normalise by average size of target set.
    @printf(
        "Loss: %8.3f +- %7.3f (%d batches; normalised)\n",
        _mean_error(values ./ mean(target_sizes))...,
        num_batches
    )

    # Normalise by the target set size.
    @printf(
        "Loss: %8.3f +- %7.3f (%d batches; global mean)\n",
        _mean_error(values ./ target_sizes)...,
        num_batches
    )

    return loss_value, loss_error
end

_mean_error(xs) = (mean(xs), 2std(xs) / sqrt(length(xs)))

_nanreport = Flux.throttle(() -> println("Encountered NaN loss! Returning zero."), 1)

function nansafe(loss, xs...)
    value = loss(xs...)
    if isnan(value)
        _nanreport()
        return Tracker.track(identity, 0f0)
    else
        return value
    end
end

function train!(
    model,
    loss,
    data_gen,
    opt;
    bson=nothing,
    starting_epoch=1,
    epochs=100,
    tasks_per_epoch=1000,
    path="output"
)
    GPUArrays.allowscalar(false)

    # Divide out batch size to get the number of batches per epoch.
    batches_per_epoch = div(tasks_per_epoch, data_gen.batch_size)

    # Display the settings of the training run.
    @printf("Epochs:               %-6d\n", epochs)
    @printf("Starting epoch:       %-6d\n", starting_epoch)
    @printf("Tasks per epoch:      %-6d\n", batches_per_epoch * data_gen.batch_size)
    @printf("Batch size:           %-6d\n", data_gen.batch_size)
    @printf("Batches per epoch:    %-6d\n", batches_per_epoch)

    # Evaluate once before training.
    eval_model(model, loss, data_gen, 1)

    for epoch in starting_epoch:(starting_epoch + epochs - 1)
        # Perform epoch.
        println("Epoch: $epoch")
        @time Flux.train!(
            (xs...) -> nansafe(loss, model, epoch, gpu.(xs)...),
            Flux.params(model),
            data_gen(batches_per_epoch),
            opt
        )

        # Evalute model.
        loss_value, loss_error = eval_model(model, loss, data_gen, epoch)
        plot_task(model, data_gen, epoch, make_plot_true(data_gen.process), path=path)

        if !isnothing(bson)
            checkpoint!(bson, model, epoch, loss_value, loss_error)
        end
    end
end

function report_num_params(model)
    @printf("Number of parameters: %-6d\n", sum(map(length, Flux.params(model))))
end

function plot_task(
    model,
    data_gen,
    epoch,
    plot_true = (plt, xc, yc, xt) -> nothing;
    path = "output"
)
    x = collect(range(-3, 3, length=400))

    # Predict on a task.
    xc, yc, xt, yt = map(x -> x[:, 1, 1], data_gen(1)[1])
    μ, lower, upper, samples = predict(model, xc, yc, x)

    plt = plot()

    # Scatter target and context set.
    scatter!(plt, xt, yt, c=:red, label="Target set", dpi=200)
    scatter!(plt, xc, yc, c=:black, label="Context set", dpi=200)

    # Plot prediction of true, underlying model.
    plot_true(plt, xc, yc, x)

    # Plot prediction.
    if !isnothing(μ)
        plot!(plt, x, μ, c=:green, label="Model output", dpi=200)
        plot!(
            plt,
            x,
            [μ μ],
            fillrange=[lower upper],
            fillalpha=0.2,
            c=:green,
            label="",
            dpi=200
        )
    end

    # Plot samples.
    if !isnothing(samples)
        plot!(plt, x, samples, c=:green, lw=0.5, dpi=200, label="")
    end

    if !isnothing(path)
        savefig(plt, "$path/epoch$epoch.png")
    end
end

make_plot_true(process) = (plt, xc, yc, xt) -> nothing

function make_plot_true(process::GP)
    function plot_true(plt, xc, yc, xt)
        xc = Float64.(xc)
        yc = Float64.(yc)
        xt = Float64.(xt)
        posterior = process | Obs(process(xc, 1e-6) ← yc)
        margs = marginals(posterior(xt))
        plot!(plt, xt, mean.(margs), c=:blue, label="GP", dpi=200)
        plot!(
            plt,
            xt,
            mean.(margs) .- 2 .* std.(margs),
            c=:blue,
            linestyle=:dash,
            label="",
            dpi=200
        )
        plot!(
            plt,
            xt,
            mean.(margs) .+ 2 .* std.(margs),
            c=:blue,
            linestyle=:dash,
            label="",
            dpi=200
        )
    end
    return plot_true
end

end
