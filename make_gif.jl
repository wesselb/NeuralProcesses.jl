using BSON
using NeuralProcesses
using Flux
using Random
using Plots
using Printf

pyplot()

# Load existing model.
@BSON.load "sawtooth.bson" model_cpu

function predict(x_context, y_context, x_target)
    expand(x) = reshape(x, length(x), 1, 1)
    y_mean, y_var = map(
        x -> Flux.data(x[:, 1, 1]),
        model_cpu(expand.((x_context, y_context, x))...)
    )
    return y_mean, y_mean .- 2 .* sqrt.(y_var), y_mean .+ 2 .* sqrt.(y_var)
end

s = Sawtooth()
x = collect(range(-2, 2, length=501))
y = rand(s(x, 1e-10))

inds = randperm(length(x))
x_context = x[inds]
y_context = y[inds]

for num = 0:15
    y_mean, y_lower, y_upper = predict(x_context[1:num], y_context[1:num], x)

    plt = plot()

    # Plot true sawtooth.
    plot!(plt, x, y, c=:black, dpi=200, label="")

    # Plot prediction.
    plot!(plt, x, y_mean, c=:green, label="", dpi=200)
    plot!(plt, x, [y_mean y_mean],
        fillrange=[y_lower, y_upper],
        fillalpha=0.2,
        c=:green,
        label="",
        dpi=200
    )

    # Plot context set.
    scatter!(
        plt,
        x_context[1:num],
        y_context[1:num],
        color=:red,
        markerstrokecolor=:red,
        markersize=5,
        label="",
        dpi=200
    )

    ylims!(plt, (-0.25, 1.25))
    xlims!(plt, (-2, 2))
    plot!(plt, framestyle=:none, size=(1000, 200))

    savefig(plt, @sprintf("step%02d.png", num))
end

run(`convert -delay 50 -loop 0 step*.png loop.gif`)
for num = 0:15
    rm(@sprintf("step%02d.png", num))
end
