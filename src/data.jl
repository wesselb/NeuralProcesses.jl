export DataGenerator, Sawtooth

"""
    DataGenerator

# Fields
- `process`: Something that can be called at inputs `x` and a noise level `noise` and gives
    back a distribution that can be fed to `randn` to sample values corresponding to those
    inputs `x` at observation noise `noise`.
- `batch_size::Integer=16`: Number of tasks in a batch.
- `x_dist::Distribution=Uniform(-2, 2)`: Distribution to sample inputs from.
- `max_context_points::Integer=10`: Maximum number of context points in a task.
- `num_target_points::Integer=100`: Number of target points in a task.
"""
struct DataGenerator
    process
    batch_size::Integer
    x_dist::Distribution
    max_context_points::Integer
    num_target_points::Integer

    function DataGenerator(
        process;
        batch_size::Integer=16,
        x_dist::Distribution=Uniform(-2, 2),
        max_context_points::Integer=10,
        num_target_points::Integer=100,
    )
        return new(
            process,
            batch_size,
            x_dist,
            max_context_points,
            num_target_points
        )
    end
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

_float32(x) = Float32.(x)

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

        push!(tasks, _float32.((
            x_context,
            y[1:num_context],
            x_target,
            y[num_context + 1:end]
        )))
    end

    # Collect as a batch and return.
    return map(x -> cat(x...; dims=3), zip(tasks...))
end

"""
    Sawtooth

Random truncated Fourier expansion of a sawtooth wave.

# Fields
- `freq_dist=Uniform(3, 5)`: Distribution of the frequency.
- `shift_dist=Uniform(3, 5)`: Distribution of the shift.
- `trunc_dist=10:20`: Distribution of the truncation of the Fourier expansion.
"""
struct Sawtooth
    freq_dist
    shift_dist
    trunc_dist

    function Sawtooth(
        freq_dist=Uniform(3, 5),
        shift_dist=Uniform(-5, 5),
        trunc_dist=10:20
    )
        return new(freq_dist, shift_dist, trunc_dist)
    end
end

"""
    FiniteSawtooth{T<:Real}

Finite-dimensional distribution of a `Sawtooth` at particular inputs.

# Fields
- `x::Vector{T}`: Inputs.
- `noise::T`: Noise variance.
- `sawtooth::Sawtooth`: Corresponding sawtooth.
"""
struct FiniteSawtooth{T<:Real}
    x::Vector{T}
    noise::T
    sawtooth::Sawtooth
end

"""
    (s::Sawtooth)(x, noise)

Construct the finite-dimensional distribution of a `Sawtooth` at inputs `x` and observation
noise variance `noise`.

# Arguments
- `x`: Inputs.
- `noise`: Noise variance.

# Returns
- `FiniteSawtooth`: Corresponding finite-dimensional distribution.
"""
(s::Sawtooth)(x, noise) = FiniteSawtooth(x, noise, s)

function Base.rand(fs::FiniteSawtooth)
    # Sample parameters for particular sawtooth wave.
    amp = 1
    freq = rand(fs.sawtooth.freq_dist)
    shift = rand(fs.sawtooth.shift_dist)
    trunc = rand(fs.sawtooth.trunc_dist)

    # Apply shift.
    x = fs.x .+ shift

    # Construct expansion.
    k = collect(range(1, trunc + 1, step=1))'
    f = 0.5amp .- (amp / pi) .* sum((-1).^k .* sin.(2pi .* k .* freq .* x) ./ k, dims=2)

    # Add noise and return.
    return f .+ sqrt(fs.noise) .* randn(eltype(x), size(x)...)
end
