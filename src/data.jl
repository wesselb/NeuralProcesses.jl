export DataGenerator

"""
    DataGenerator

# Fields
- `process`: Something that can be called at inputs `x` and a noise level `noise` and gives
    back a distribution that can be fed to `randn` to sample values corresponding to those
    inputs `x` at observation noise `noise`.
- `batch_size::Integer`: Number of tasks in a batch.
- `x_dist::Distribution`: Distribution to sample inputs from.
- `max_context_points::Integer`: Maximum number of context points in a task.
- `num_target_points::Integer`: Number of target points in a task.
"""
struct DataGenerator
    process
    batch_size::Integer
    x_dist::Distribution
    max_context_points::Integer
    num_target_points::Integer
end

"""
    DataGenerator(
        k::Stheno.Kernel;
        batch_size::Integer=16,
        x_dist::Distribution=Uniform(-2, 2),
        max_context_points::Integer=10,
        num_target_points::Integer=100,
    )

# Arguments
- `k::Stheno.Kernel`: Kernel of a Gaussian process to sample from.

# Fields
- `batch_size::Integer`: Number of tasks in a batch.
- `x_dist::Distribution`: Distribution to sample inputs from.
- `max_context_points::Integer`: Maximum number of context points in a task.
- `num_target_points::Integer`: Number of target points in a task.
"""
function DataGenerator(
    k::Stheno.Kernel;
    batch_size::Integer=16,
    x_dist::Distribution=Uniform(-2, 2),
    max_context_points::Integer=10,
    num_target_points::Integer=100,
)
    gp = GP(k, GPC())
    return DataGenerator(
        gp,
        batch_size,
        x_dist,
        max_context_points,
        num_target_points
    )
end

"""
    (generator::DataGenerator)(num_batches::Integer)

# Arguments
- `num_batches::Integer`: Number of batches to sample.

# Returns
- Array of named tuple with fields `x_context`, `y_context`, `x_target`, and `y_target`,
    in that order.
"""
function (generator::DataGenerator)(num_batches::Integer)
    return [_make_batch(
        generator,
        rand(0:generator.max_context_points),
        generator.num_target_points
    ) for i in 1:num_batches]
end

_float32_gpu(x) = gpu(Float32.(x))

function _make_batch(generator::DataGenerator, num_context::Integer, num_target::Integer)
    # Sample tasks.
    tasks = []
    for i in 1:generator.batch_size
        # Determine context set.
        x_context = rand(generator.x_dist, num_context)

        # Determine target set.
        dx = (maximum(generator.x_dist) - minimum(generator.x_dist)) / num_target
        offset = rand() * dx
        steps = collect(range(0, num_target - 1, step=1))
        x_target = minimum(generator.x_dist) .+ offset .+ steps .* dx

        # Concatenate inputs and sample.
        x = vcat(x_context, x_target)
        y = rand(generator.process(x, 1e-10))

        push!(tasks, _float32_gpu.((
            x_context,
            y[1:num_context],
            x_target,
            y[num_context + 1:end]
        )))
    end

    # Collect as a batch and return.
    batch = map(x -> cat(x...; dims=3), zip(tasks...))
    return (
        x_context=batch[1],
        y_context=batch[2],
        x_target=batch[3],
        y_target=batch[4]
    )
end
