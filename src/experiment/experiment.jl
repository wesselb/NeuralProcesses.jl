module Experiment

export predict, loss, eval_model, train!, plot_task

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
    values = map(
        x -> loss(model, epoch, gpu.(x)...),
        data_gen(num_batches)
    )
    loss_value = mean(values)
    loss_error = 2std(values) / sqrt(length(values))
    @printf(
        "Loss: %.3f +- %.3f (%d batches)\n",
        loss_value,
        loss_error,
        num_batches
    )
    return loss_value, loss_error
end

function nansafe(loss, xs...)
    value = loss(xs...)
    if isnan(value)
        println("Encountered NaN loss! Returning zero.")
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
    batches_per_epoch=2048,
    path="output"
)
    GPUArrays.allowscalar(false)

    # Evaluate once before training.
    eval_model(model, loss, data_gen, 1)

    for epoch in starting_epoch:(starting_epoch + epochs - 1)
        # Perform epoch.
        println("Epoch: $epoch")
        Flux.train!(
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

function plot_task(
    model,
    data_gen,
    epoch,
    plot_true = (plt, x_context, y_context, x_target) -> nothing;
    path = "output"
)
    x = collect(range(-3, 3, length=400))

    # Predict on a task.
    x_context, y_context, x_target, y_target = map(x -> x[:, 1, 1], data_gen(1)[1])
    μ, lower, upper, samples = predict(model, x_context, y_context, x)

    plt = plot()

    # Scatter target and context set.
    scatter!(plt, x_target, y_target, c=:red, label="Target set", dpi=200)
    scatter!(plt, x_context, y_context, c=:black, label="Context set", dpi=200)

    # Plot prediction of true, underlying model.
    plot_true(plt, x_context, y_context, x)

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

make_plot_true(process) = (plt, x_context, y_context, x_target) -> nothing

function make_plot_true(process::GP)
    function plot_true(plt, x_context, y_context, x_target)
        x_context = Float64.(x_context)
        y_context = Float64.(y_context)
        x_target = Float64.(x_target)
        posterior = process | Obs(process(x_context, 1e-10) ← y_context)
        margs = marginals(posterior(x_target))
        plot!(plt, x_target, mean.(margs), c=:blue, label="GP", dpi=200)
        plot!(
            plt,
            x_target,
            mean.(margs) .- 2 .* std.(margs),
            c=:blue,
            linestyle=:dash,
            label="",
            dpi=200
        )
        plot!(
            plt,
            x_target,
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
