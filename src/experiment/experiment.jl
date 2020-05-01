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

include("checkpoint.jl")

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

function kl(μ₁, σ²₁, μ₂, σ²₂)
    return (log.(σ²₂ ./ σ²₁) .+ (σ²₁ .+ (μ₁ .- μ₂).^2) ./ σ²₂ .- 1) ./ 2
end

function loss(model::ConvNP, epoch, x_context, y_context, x_target, y_target, num_samples)
    x_latent = model.disc(x_context, x_target) |> gpu

    pz = encode(model, x_context, y_context, x_latent)
    qz = encode(
        model,
        cat(x_context, x_target, dims=1),
        cat(y_context, y_target, dims=1),
        x_latent
    )

    samples = sample_latent(model, qz..., num_samples)
    μ, σ² = decode(model, x_latent, samples, x_target)

    # Compute the components of the ELBO.
    expectations = gaussian_logpdf(y_target, μ, σ²)
    kls = kl(qz..., pz...)

    # Sum over data points and channels to assemble the expressions.
    expectations = sum(expectations, dims=(1, 3))
    kls = sum(kls, dims=(1, 3))

    # Estimate ELBO from samples.
    elbos = mean(expectations, dims=5) .- kls

    # Return average over batches.
    return -mean(elbos)
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

    return -logpdf / batch_size
end

function eval_model(model, data_gen, epoch; num_batches=256, loss_args=())
    model = _untrack(model)
    values = map(
        x -> loss(model, epoch, gpu.(x)..., loss_args...),
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
    path="output",
    loss_args=()
)
    GPUArrays.allowscalar(false)

    # Evaluate once before training.
    eval_model(model, data_gen, 1, loss_args=loss_args)

    for epoch in starting_epoch:(starting_epoch + epochs - 1)
        # Perform epoch.
        println("Epoch: $epoch")
        Flux.train!(
            (xs...) -> loss(model, epoch, gpu.(xs)..., loss_args...),
            Flux.params(model),
            data_gen(batches_per_epoch),
            opt
        )

        # Evalute model.
        loss_value, loss_error = eval_model(model, data_gen, epoch, loss_args=loss_args)
        plot_task(model, data_gen, epoch, make_plot_true(data_gen.process), path=path)

        if !isnothing(bson)
            checkpoint(bson, model, epoch, loss_value, loss_error)
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
