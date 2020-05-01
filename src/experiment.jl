module Experiment

export predict, loss, eval_model, train!, plot_task

using ..ConvCNPs

using BSON
using Flux
using Stheno
using StatsBase
using Plots
using Printf
using GPUArrays

pyplot()

_untrack(model) = mapleaves(x -> Flux.data(x), model)

_expand_gpu(x) = gpu(reshape(x, length(x), 1, 1))

function predict(
    model::ConvCNP,
    x_context::AbstractVector,
    y_context::AbstractVector,
    x_target::AbstractVector
)
    μ, σ² = _untrack(model)(_expand_gpu.((x_context, y_context, x_target))...)
    μ = μ[:, 1, 1] |> cpu
    σ² = σ²[:, 1, 1] |> cpu
    return μ, μ .- 2 .* sqrt.(σ²), μ .+ 2 .* sqrt.(σ²), nothing
end

function predict(
    model::CorrelatedConvCNP,
    x_context::AbstractVector,
    y_context::AbstractVector,
    x_target::AbstractVector
)
    μ, Σ = _untrack(model)(_expand_gpu.((x_context, y_context, x_target)))
    μ = μ[:, 1, 1] |> cpu
    Σ = Σ[:, :, 1] |> cpu
    σ² = diag(Σ)

    # Produce three posterior samples.
    samples = cholesky(y_cov).U' * randn(length(x), 3) .+ y_mean

    return μ, μ .- 2 .* sqrt.(σ²), μ .+ 2 .* sqrt.(σ²), samples
end

function loss(model::ConvCNP, epoch, x_context, y_context, x_target, y_target)
    logpdfs = gaussian_logpdf(y_target, model(x_context, y_context, x_target)...)
    # Sum over data points before averaging over tasks.
    return -mean(sum(logpdfs, dims=1))
end

_epoch_to_reg(epoch) = 10^(-min(1 + Float32(epoch), 5))

function loss(model::CorrelatedConvCNP, epoch, x_context, y_context, x_target, y_target)
    size(y_target, 2) == 1 || error("Target outputs have more than one channel.")

    n_target, _, batch_size = size(x_target)

    μ, Σ = model(x_context, y_context, x_target)

    logpdf = 0f0
    ridge = gpu(Matrix(_epoch_to_reg(epoch) * I, n_target, n_target))
    for i = 1:batch_size
        logpdf += gaussian_logpdf(y_target[:, 1, i], μ[:, i], Σ[:, :, i] .+ ridge)
    end

    return -logpdf / n_target / batch_size
end

function eval_model(model, data_gen, epoch; num_batches=128)
    model = _untrack(model)
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

function train!(
    model,
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
    eval_model(model, data_gen, 1)

    for epoch in starting_epoch:(starting_epoch + epochs - 1)
        # Perform epoch.
        println("Epoch: $epoch")
        Flux.train!(
            (xs...) -> loss(model, epoch, gpu.(xs)...),
            Flux.params(model),
            data_gen(batches_per_epoch),
            opt
        )

        # Evalute model.
        loss_value, loss_error = eval_model(model, data_gen, epoch)
        plot_task(model, data_gen, epoch, make_plot_true(data_gen.process), path=path)

        if !isnothing(bson)
            # Check whether to save model.
            save_model = false
            if !isfile(bson) || epoch == 1
                # It is the first model. Save in any case.
                println("Saving model: first model")
                save_model = true
            else
                # BSON file exists. Check whether it has a loss saved.
                content = BSON.load(bson)
                if haskey(content, :loss_value)
                    # A loss is available. Only save if the current loss is lower.
                    if loss_value < content[:loss_value]
                        println("Saving model: new best model")
                        save_model = true
                    end
                else
                    # There is no loss available. Save anyway.
                    println("Saving model: no existing loss")
                    save_model = true
                end
            end

            if save_model
                BSON.bson(
                    bson,
                    model = cpu(model),
                    loss_value = loss_value,
                    loss_error = loss_error,
                    epoch = epoch
                )
            end
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
    if !isnothing(samples)
        # Plot samples.
        plot!(plt, x, samples, c=:green, lw=0.5, dpi=200, label="")
    end

    savefig(plt, "$path/epoch$epoch.png")
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
