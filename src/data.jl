export DataGenerator, UniformUnion, Sawtooth, BayesianConvNP, Mixture

"""
    DataGenerator

# Fields
- `process`: Something that can be called at inputs `x` and a noise level `noise` and gives
    back a distribution that can be fed to `randn` to sample values corresponding to those
    inputs `x` at observation noise `noise`.
- `batch_size::Integer=16`: Number of tasks in a batch.
- `x_context::Distribution=Uniform(-2, 2)`: Distribution to sample context inputs from.
- `x_target::Distribution=Uniform(-2, 2)`: Distribution to sample target inputs from.
- `num_context::Distribution=DiscreteUniform(10, 10)`: Distribution of number of context
    points in a task.
- `num_target::Distribution=DiscreteUniform(100, 100)`: Distribution of number of target
    points in a task.
- `σ²::Float64`: Noise variance to add to the data.
"""
struct DataGenerator
    process
    batch_size::Integer
    x_context::Distribution
    x_target::Distribution
    num_context::Distribution
    num_target::Distribution
    σ²::Float64

    function DataGenerator(
        process;
        batch_size::Integer=16,
        x_context::Distribution=Uniform(-2, 2),
        x_target::Distribution=Uniform(-2, 2),
        num_context::Distribution=DiscreteUniform(10, 10),
        num_target::Distribution=DiscreteUniform(100, 100),
        σ²::Float64=1e-8
    )
        return new(
            process,
            batch_size,
            x_context,
            x_target,
            num_context,
            num_target,
            σ²
        )
    end
end

"""
    struct UniformUnion{T<:Real} <: ContinuousUnivariateDistribution

A union of various `Uniform`s.

# Fields
- `uniforms::Vector{Uniform{T}}`: Underlying `Uniforms`s.
- `probs::ProbabilityWeights`: Probabilities associated to every `Uniform`.
"""
struct UniformUnion{T<:Real} <: ContinuousUnivariateDistribution
    uniforms::Vector{Uniform{T}}
    probs::ProbabilityWeights
end

"""
    UniformUnion(uniforms::Uniform...)

Construct a `UnionUniform` and weight the uniforms according to the size of their domain.

# Arguments
- `uniforms::Uniform...`: Underlying `Uniform`s.

# Returns
- `UniformUnion`: Associated `UniformUnion`.
"""
function UniformUnion(uniforms::Uniform...)
    lengths = [maximum(d) - minimum(d) for d in uniforms]
    return UniformUnion(collect(uniforms), pweights([x / sum(lengths) for x in lengths]))
end

Base.rand(u::UniformUnion) = rand(StatsBase.sample(u.uniforms, u.probs))
Base.rand(u::UniformUnion, n::Int64) = [rand(u) for _ = 1:n]

"""
    (generator::DataGenerator)(num_batches::Integer)

# Arguments
- `num_batches::Integer`: Number of batches to sample.

# Returns
- `Vector`: Vector of tasks.
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
        xc = rand(generator.x_context, num_context)
        xt = rand(generator.x_target, num_target)

        # Concatenate inputs and sample.
        x = vcat(xc, xt)
        y = rand(generator.process(x, generator.σ²))

        push!(tasks, _float32.((
            xc,
            y[1:num_context],
            xt,
            y[num_context + 1:end]
        )))
    end

    # Collect as a batch and return.
    return map(x -> cat(x...; dims=3), zip(tasks...))
end


"""
    FDD{T}

Finite-dimensional distribution of a process at particular inputs.

# Fields
- `x`: Inputs.
- `σ²`: Noise variance.
- `process::T`: Underlying process.
"""
struct FDD{T}
    x
    σ²
    process::T
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

(s::Sawtooth)(x, noise) = FDD(x, noise, s)

function Base.rand(s::FDD{Sawtooth})
    # Sample parameters for particular sawtooth wave.
    amp = 1
    freq = rand(s.process.freq_dist)
    shift = rand(s.process.shift_dist)
    trunc = rand(s.process.trunc_dist)

    # Apply shift.
    x = s.x .- shift

    # Construct expansion.
    k = collect(range(1, trunc + 1, step=1))'
    f = 0.5amp .- (amp / pi) .* sum((-1).^k .* sin.(2pi .* k .* freq .* x) ./ k, dims=2)

    # Add noise and return.
    return f .+ sqrt(s.σ²) .* randn(eltype(x), size(x)...)
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
     `UniformDiscretisation1D`.
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

(convnp::BayesianConvNP)(x, noise) = FDD(x, noise, convnp)

function Base.rand(convnp::FDD{BayesianConvNP})
    # Contruct discretisation.
    disc = UniformDiscretisation1D(
        convnp.process.points_per_unit,
        convnp.process.receptive_field / 2,
        1
    )
    xz = reshape(disc(convnp.x), :, 1, 1)

    # Construct CNN with random initialisation.
    conv = build_conv(
        convnp.process.receptive_field,
        convnp.process.num_layers,
        convnp.process.num_channels;
        points_per_unit=convnp.process.points_per_unit,
        num_in_channels=1,
        num_out_channels=1,
        dimensionality=1,
        init_conv=_init_conv_random_bias,
        init_depthwiseconv=_init_depthwiseconv_random_bias
    )

    # Construct decoder.
    scale = 2 / convnp.process.points_per_unit
    decoder = set_conv(1, scale)

    # Draw random encoding.
    encoding = randn(Float32, length(xz))

    # Pass through CNN, which takes in images of height one.
    encoding = reshape(encoding, length(encoding), 1, 1, 1)
    latent = conv(encoding)

    # Perform decoding.
    xt = reshape(convnp.x, length(convnp.x), 1, 1)
    sample = code(decoder, xz, latent, xt)[2][:, 1, 1]  # Also returns inputs.

    # Normalise sample.
    sample = (sample .- mean(sample)) ./ std(sample)

    # Return with noise.
    return sample .+ sqrt(convnp.σ²) .* randn(Float32, size(sample)...)
end

"""
    struct Mixture

Mixture of processes.

# Fields
- `processes`: Processes in the mixture.
"""
struct Mixture
    processes
end

Mixture(processes...) = Mixture(processes)

(mixture::Mixture)(x, noise) = FDD(x, noise, mixture)

Base.rand(fmixture::FDD{Mixture}) =
    rand(rand(fmixture.process.processes)(fmixture.x, fmixture.σ²))
