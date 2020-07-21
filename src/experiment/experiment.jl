module Experiment

export predict, loss, eval_model, train!, report_num_params, plot_task

using ..NeuralProcesses

using BSON
using CUDA
using Flux
using PyPlot
using Printf
using Stheno
using Tracker

import StatsBase: std

include("checkpoint.jl")

function eval_model(model, loss, data_gen, epoch; num_batches=256)
    model = NeuralProcesses.untrack(model)
    @time tuples = map(x -> loss(model, epoch, gpu.(x)...), data_gen(num_batches))
    values = map(x -> x[1], tuples)
    sizes = map(x -> x[2], tuples)

    # Compute and print loss.
    loss_value, loss_error = _mean_error(values)
    println("Losses:")
    @printf(
        "    %8.3f +- %7.3f (%d batches)\n",
        loss_value,
        loss_error,
        num_batches
    )

    # Normalise by average size of target set.
    @printf(
        "    %8.3f +- %7.3f (%d batches; normalised)\n",
        _mean_error(values ./ mean(sizes))...,
        num_batches
    )

    # Normalise by the target set size.
    @printf(
        "    %8.3f +- %7.3f (%d batches; global mean)\n",
        _mean_error(values ./ sizes)...,
        num_batches
    )

    return loss_value, loss_error
end

_mean_error(xs) = (mean(xs), 2std(xs) / sqrt(length(xs)))

_nanreport = Flux.throttle(() -> println("Encountered NaN loss! Returning zero."), 1)

function nansafe(loss, xs...)
    value, value_size = loss(xs...)
    if isnan(value)
        _nanreport()
        return Tracker.track(identity, 0f0), value_size
    else
        return value, value_size
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
    CUDA.GPUArrays.allowscalar(false)

    # Divide out batch size to get the number of batches per epoch.
    batches_per_epoch = div(tasks_per_epoch, data_gen.batch_size)

    # Display the settings of the training run.
    @printf("Epochs:               %-6d\n", epochs)
    @printf("Starting epoch:       %-6d\n", starting_epoch)
    @printf("Tasks per epoch:      %-6d\n", batches_per_epoch * data_gen.batch_size)
    @printf("Batch size:           %-6d\n", data_gen.batch_size)
    @printf("Batches per epoch:    %-6d\n", batches_per_epoch)

    # Track the parameters of the model for training.
    model = NeuralProcesses.track(model)

    for epoch in starting_epoch:(starting_epoch + epochs - 1)
        # Perform epoch.
        println("Epoch: $epoch")
        @time begin
            ps = Flux.Params(Flux.params(model))
            for d in data_gen(batches_per_epoch)
                gs = Tracker.gradient(ps) do
                    first(nansafe(loss, model, epoch, gpu.(d)...))
                end
                for p in ps
                    Tracker.update!(p, -Flux.Optimise.apply!(opt, p, Tracker.data(gs[p])))
                end
            end
        end

        # Evalute model.
        loss_value, loss_error = eval_model(
            NeuralProcesses.untrack(model),
            loss,
            data_gen,
            epoch
        )

        # Plot model.
        plot_task(
            NeuralProcesses.untrack(model),
            data_gen,
            epoch,
            make_plot_true(data_gen.process),
            path=path
        )

        # Save result.
        if !isnothing(bson)
            checkpoint!(
                bson,
                NeuralProcesses.untrack(model),
                epoch,
                loss_value,
                loss_error
            )
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
    plot_true = (plt, xc, yc, xt, σ²) -> nothing;
    path = "output",
    num_tasks = 5
)
    for i = 1:num_tasks
        x = collect(range(-3, 3, length=400))

        # Predict on a task.
        xc, yc, xt, yt = map(x -> x[:, 1, 1], data_gen(1)[1])
        μ, lower, upper, samples = predict(model, xc, yc, x)

        figure(figsize=(10,6))

        # Scatter target and context set.
        scatter(xt, yt, c="r", label="Target set")
        scatter(xc, yc, c="b", label="Context set")

        # Plot prediction of true, underlying model.
        plot_true(xc, yc, x, data_gen.σ²)

        # Plot prediction.
        if !isnothing(μ)
            plot(x, μ, c="g", label="Model output")
            fill_between(x, lower, upper, fillalpha=0.2, facecolor=:green)
        end

        # Plot samples.
        if !isnothing(samples)
            plot(x, samples, c="g", lw=0.5)
        end

        if !isnothing(path)
            savefig("$path/epoch$epoch-$i.png")
        end

        close()
    end
end

make_plot_true(process) = (plt, xc, yc, xt, σ²) -> nothing

function make_plot_true(process::GP)
    function plot_true(plt, xc, yc, xt, σ²)
        xc = Float64.(xc)
        yc = Float64.(yc)
        xt = Float64.(xt)
        posterior = process | Obs(process(xc, σ²) ← yc)
        margs = marginals(posterior(xt))
        plot(xt, mean.(margs), c="b", label="GP")
        error = 2 .* sqrt.(std.(margs).^2 .+ σ²)
        plot(xt, mean.(margs) .- error, c="b", ls="--")
        plot(xt, mean.(margs) .+ error, c="b", ls="--")
    end
    return plot_true
end

end
