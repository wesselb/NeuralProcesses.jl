module Experiment

export predict, loss, eval_model, train!, report_num_params, plot_task

using ..NeuralProcesses

using BSON
using CUDA
using Flux
using PyPlot
using Printf
using ProgressMeter
using Stheno
using Tracker

import StatsBase: std

include("checkpoint.jl")

"""
    eval_model(model, loss, data_gen, epoch::Integer; num_batches::Integer=256)

Evaluate model.

# Arguments
- `model`: Model to evaluate.
- `loss`: Loss function.
- `data_gen`: Data generator.
- `epoch::Integer`: Current epoch.

# Keywords
- `num_batches::Integer=256`: Number of batches to use.

# Returns
- `Tuple{Float32, Float32}`: Loss value and error.
"""
function eval_model(model, loss, data_gen, epoch::Integer; num_batches::Integer=256)
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

function _nansafe(loss, xs...)
    value, value_size = loss(xs...)
    if isnan(value)
        _nanreport()
        return Tracker.track(identity, 0f0), value_size
    else
        return value, value_size
    end
end

"""
    train!(
        model,
        loss,
        data_gen,
        opt;
        bson=nothing,
        starting_epoch::Integer=1,
        epochs::Integer=100,
        tasks_per_epoch::Integer=1000,
        path="output"
    )

Train a model.

# Arguments
- `model`: Model to train.
- `loss`: Loss function.
- `data_gen`: Data generator.
- `opt`: Optimiser. See `Flux.Optimiser`.

# Keywords
- `bson`: Name of file to save model to. Set to `nothing` to not save the model.
- `starting_epoch::Integer=1`: Epoch to start training at.
- `epochs::Integer=100`: Number of epochs to train for.
- `tasks_per_epoch::Integer=1000`: Number of tasks to draw in each poch.
- `path="output"`: The model will be tested after every epochs, which produces plots. This
    specifies the directory to write plots to.
"""
function train!(
    model,
    loss,
    data_gen,
    opt;
    bson=nothing,
    starting_epoch::Integer=1,
    epochs::Integer=100,
    tasks_per_epoch::Integer=1000,
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
        CUDA.reclaim()
        @time begin
            ps = Flux.Params(Flux.params(model))
            @showprogress "Epoch $epoch: " for d in data_gen(batches_per_epoch)
                gs = Tracker.gradient(ps) do
                    first(_nansafe(loss, model, epoch, gpu.(d)...))
                end
                for p in ps
                    Tracker.update!(p, -Flux.Optimise.apply!(opt, p, Tracker.data(gs[p])))
                end
            end
        end

        # Evalute model.
        CUDA.reclaim()
        loss_value, loss_error = eval_model(
            NeuralProcesses.untrack(model),
            loss,
            data_gen,
            epoch
        )

        # Plot model.
        CUDA.reclaim()
        _plot_task(
            NeuralProcesses.untrack(model),
            data_gen,
            epoch,
            _make_plot_true(data_gen.process),
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

"""
    report_num_params(model)

Report the number of parameters of a model.

# Arguments
- `model`: Model to report the number of parameters of.
"""
function report_num_params(model)
    @printf("Number of parameters: %-6d\n", sum(map(length, Flux.params(model))))
end

function _plot_task(
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
        μ, lower, upper, samples = predict(model, xc, yc, x, epoch=epoch)

        figure(figsize=(8, 4))

        # Scatter target and context set.
        scatter(xt, yt, c="tab:red", label="Target set")
        scatter(xc, yc, c="black", label="Context set")

        # Plot prediction of true, underlying model.
        plot_true(xc, yc, x, data_gen.σ²)

        # Plot prediction.
        if !isnothing(μ)
            plot(x, μ, c="tab:green", label="Model output")
            fill_between(x, lower, upper, alpha=0.2, facecolor="tab:green")
        end

        # Plot samples.
        if !isnothing(samples)
            plot(x, samples, c="tab:green", lw=0.5)
        end

        legend()
        tight_layout()

        if !isnothing(path)
            savefig("$path/epoch$epoch-$i.png", dpi=200)
        end

        close()
    end
end

_make_plot_true(process) = (xc, yc, xt, σ²) -> nothing

function _make_plot_true(process::GP)
    function plot_true(xc, yc, xt, σ²)
        xc, yc, xt = Float64.(xc), Float64.(yc), Float64.(xt)
        posterior = process | Obs(process(xc, σ²) ← yc)
        margs = marginals(posterior(xt))
        plot(xt, mean.(margs), c="tab:blue", label="GP")
        error = 2 .* sqrt.(std.(margs).^2 .+ σ²)
        plot(xt, mean.(margs) .- error, c="tab:blue", ls="--")
        plot(xt, mean.(margs) .+ error, c="tab:blue", ls="--")
    end
    return plot_true
end

end
