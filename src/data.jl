export DataGenerator, Sawtooth, BayesianConvNP

"""
    DataGenerator

# Fields
- `process`: Something that can be called at inputs `x` and a noise level `noise` and gives
    back a distribution that can be fed to `randn` to sample values corresponding to those
    inputs `x` at observation noise `noise`.
- `batch_size::Integer=16`: Number of tasks in a batch.
- `x::Distribution=Uniform(-2, 2)`: Distribution to sample inputs from.
- `num_context::Distribution=DiscreteUniform(10, 10)`: Distribution of number of context
    points in a task.
- `num_target::Distribution=DiscreteUniform(100, 100)`: Distribution of number of target
    points in a task.
"""
struct DataGenerator
    process
    batch_size::Integer
    x::Distribution
    num_context::Distribution
    num_target::Distribution

    function DataGenerator(
        process;
        batch_size::Integer=16,
        x::Distribution=Uniform(-2, 2),
        num_context::Distribution=DiscreteUniform(10, 10),
        num_target::Distribution=DiscreteUniform(100, 100)
    )
        return new(
            process,
            batch_size,
            x,
            num_context,
            num_target
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
        rand(generator.num_context),
        rand(generator.num_target)
    ) for i in 1:num_batches]
end

_float32(x) = Float32.(x)

function _make_batch(generator::DataGenerator, num_context::Integer, num_target::Integer)
    # Sample tasks.
    tasks = []
    for i in 1:generator.batch_size
        # Determine context and target set.
        x_context = rand(generator.x, num_context)
        x_target = rand(generator.x, num_target)

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
- `freq_dist::Distribution=Uniform(3, 5)`: Distribution of the frequency.
- `shift_dist::Distribution=Uniform(-5, 5)`: Distribution of the shift.
- `trunc_dist::Distribution=DiscreteUniform(10, 20)`: Distribution of the truncation of
    the Fourier expansion.
"""
struct Sawtooth
    freq_dist::Distribution
    shift_dist::Distribution
    trunc_dist::Distribution

    function Sawtooth(
        freq_dist::Distribution=Uniform(3, 5),
        shift_dist::Distribution=Uniform(-5, 5),
        trunc_dist::Distribution=DiscreteUniform(10, 20)
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

"""
    Bayesian ConvNP

Bayesian ConvNP.

# Fields
- `receptive_field::Float32`: Width of the receptive field.
- `num_layers::Integer`: Number of layers of the CNN, excluding an initial
    and final pointwise convolutional layer to change the number of channels
    appropriately.
- `num_channels::Integer`: Number of channels of the CNN.
- `points_per_unit::Float32=30f0`: Points per unit for the discretisation. See
     `UniformDiscretisation1d`.
"""
struct BayesianConvNP
    receptive_field::Float32
    num_layers::Integer
    num_channels::Integer
    points_per_unit::Float32

    function BayesianConvNP(;
        receptive_field::Float32=1f0,
        num_layers::Integer=4,
        num_channels::Integer=6,
        points_per_unit::Float32=30f0
    )
        new(receptive_field, num_layers, num_channels, points_per_unit)
    end
end

"""
    FiniteBayesianConvNP{T<:Real}

Finite-dimensional distribution of a `BayesianConvNP` at particular inputs.

# Fields
- `x::Vector{T}`: Inputs.
- `noise::T`: Noise variance.
- `convnp::BayesianConvNP`: Corresponding `BayesianConvNP`.
"""
struct FiniteBayesianConvNP{T<:Real}
    x::Vector{T}
    noise::T
    convnp::BayesianConvNP
end

"""
    (convnp::BayesianConvNP)(x, noise)

Construct the finite-dimensional distribution of a `BayesianConvNP` at inputs `x` and
observation noise variance `noise`.

# Arguments
- `x`: Inputs.
- `noise`: Noise variance.

# Returns
- `FiniteBayesianConvNP`: Corresponding finite-dimensional distribution.
"""
(convnp::BayesianConvNP)(x, noise) = FiniteBayesianConvNP(x, noise, convnp)

function Base.rand(fconvnp::FiniteBayesianConvNP)
    # Contruct discretisation.
    disc = UniformDiscretisation1d(
        fconvnp.convnp.points_per_unit,
        fconvnp.convnp.receptive_field / 2,
        1
    )
    x_disc = disc(fconvnp.x)
    x_disc = reshape(x_disc, length(x_disc), 1, 1)

    # Construct CNN with random initialisation.
    conv = build_conv(
        fconvnp.convnp.receptive_field,
        fconvnp.convnp.num_layers,
        fconvnp.convnp.num_channels;
        points_per_unit=fconvnp.convnp.points_per_unit,
        in_channels=1,
        out_channels=1,
        dimensionality=1,
        init_conv=_init_conv_random_bias,
        init_depthwiseconv=_init_depthwiseconv_random_bias
    ).conv

    # Construct decoder.
    scale = 2 / fconvnp.convnp.points_per_unit
    decoder = SetConv([log(scale)])

    # Draw random encoding.
    encoding = randn(Float32, length(x_disc))

    # Pass through CNN, which takes in images of height one.
    encoding = reshape(encoding, length(encoding), 1, 1, 1)
    latent = conv(encoding)

    # Perform decoding.
    x_target = reshape(fconvnp.x, length(fconvnp.x), 1, 1)
    sample = decode(decoder, x_disc, latent, x_target)[:, 1, 1]

    # Normalise sample.
    sample = (sample .- mean(sample)) ./ std(sample)

    # Return with noise.
    return sample .+ sqrt(fconvnp.noise) .* randn(Float32, size(sample)...)
end
