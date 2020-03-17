export DataGenerator

"""
    DataGenerator

# Fields
- `process`: Something that can be called at inputs `x` and gives back a distribution that
    can be fed to `randn` to sample values corresponding to those inputs `x`.
- `batch_size::Integer`: Number of tasks in a batch.
- `x_dist::Distribution`: Distribution to sample inputs from.
- `max_context_points::Integer`: Maximum number of context points in a task.
- `max_target_points::Integer`: Maximum number of target points in a task.
"""
struct DataGenerator
    process
    batch_size::Integer
    x_dist::Distribution
    max_context_points::Integer
    max_target_points::Integer
end

"""
    DataGenerator(
        k::Stheno.Kernel;
        batch_size::Integer=16,
        x_dist::Distribution=Uniform(-2, 2),
        max_context_points::Integer=50,
        max_target_points::Integer=50,
    )

# Arguments
- `k::Stheno.Kernel`: Kernel of a Gaussian process to sample from.

# Fields
- `batch_size::Integer`: Number of tasks in a batch.
- `x_dist::Distribution`: Distribution to sample inputs from.
- `max_context_points::Integer`: Maximum number of context points in a task.
- `max_target_points::Integer`: Maximum number of target points in a task.
"""
function DataGenerator(
    k::Stheno.Kernel;
    batch_size::Integer=16,
    x_dist::Distribution=Uniform(-2, 2),
    max_context_points::Integer=50,
    max_target_points::Integer=50,
)
    gp = GP(k, GPC())
    return DataGenerator(
        x -> gp(x, 1e-10),
        batch_size,
        x_dist,
        max_context_points,
        max_target_points
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
    n_context = 3:generator.max_context_points
    n_target = 3:generator.max_target_points
    return [_make_batch(generator, rand(n_context), rand(n_target)) for i in 1:num_batches]
end

function _make_batch(generator::DataGenerator, num_context::Integer, num_target::Integer)
    # Sample tasks.
    tasks = []
    for i in 1:generator.batch_size
        x = rand(generator.x_dist, num_context + num_target)
        y = rand(generator.process(x))
        push!(tasks, (
            x[1:num_context],
            y[1:num_context],
            x[num_context + 1:end],
            y[num_context + 1:end]
        ))
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
