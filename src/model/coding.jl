export encode, decode, MultiHead, reencode_stochastic, TargetAggregator, FunctionalAggregator

encode(f, xz, z, _) = xz, f(z)
decode(f, xz, z, _) = xz, f(z)

"""
    struct AggregateEncoding

Aggregation of encodings.

# Fields
- `encodings`: Aggregated encodings.
"""
struct AggregateEncoding
    encodings
end

sample(encoding::AggregateEncoding; num_samples::Integer) =
    AggregateEncodingSample(sample.(encoding.encodings, num_samples=num_samples))

"""
    struct AggregateEncodingSample

Sample of an aggregation of encodings.

# Fields
- `samples`: Samples.
"""
struct AggregateEncodingSample
    samples
end

"""
    materialise(sample::AggregateEncodingSample)

Turn a sample of an aggregation of encodings into a tensor.

# Arguments
- `sample::AggregateEncodingSample`: Sample to turn into a tensor.

# Returns
- `AA`: Tensor corresponding to sample.
"""
materialise(sample::AggregateEncodingSample) = repeat_cat(sample.samples..., dims=2)

# Decoders typically cannot handle a fourth dimension (sample dimension) or an aggregation
# of samples. Hence, we implement a generic fallback that automatically takes care of this.

function decode(decoder, xz::AA, z::AggregateEncodingSample, x::AA)
    z = materialise(z)

    # Repeat the inputs over samples to match batch dimensions.
    num_samples = size(z, 4)
    xz = repeat_gpu(xz, 1, 1, 1, num_samples)
    x  = repeat_gpu(x, 1, 1, 1, num_samples)

    # Merge the sample and batch dimension.
    xz, back = to_rank(3, xz)
    z, _     = to_rank(3, z)
    x, _     = to_rank(3, x)

    # Perform decoding.
    x, d = decode(decoder, xz, z, x)

    return x, map(d, back)  # Separate samples from batches again.
end

function kl(encoding1::AggregateEncoding, encoding2::AggregateEncoding)
    return reduce((x, y) -> x .+ y, [
        sum(kl(d1, d2), dims=(1, 2))
        for (d1, d2) in zip(encoding1.encodings, encoding2.encodings)
    ])
end

function logpdf(encoding::AggregateEncoding, sample::AggregateEncodingSample)
    return reduce((x, y) -> x .+ y, [
        sum(logpdf(d, z), dims=(1, 2))
        for (d, z) in zip(encoding.encodings, sample.samples)
    ])
end

"""
    struct MultiHead

# Fields
- `splitter`: Function that splits the input into multiple pieces.
- `heads`: One head for every piece that `splitter` produces.
"""
struct MultiHead
    splitter
    heads
end

MultiHead(splitter, heads...) = MultiHead(splitter, heads)

@Flux.treelike MultiHead

(mh::MultiHead)(xs) = collect(zip([
    head(x) for (head, x) in zip(mh.heads, mh.splitter(xs))
]...))

decode(mh::MultiHead, xz, zs::AA, xt) = collect(zip([
    decode(head, xz, z, xt) for (head, z) in zip(mh.heads, mh.splitter(zs))
]...))

# Chains are the generic construct used to compose encoders and decoders.

function encode(chain::Chain, xz, z, x)
    for f in chain
        xz, z = encode(f, xz, z, x)
    end
    return xz, z
end

function decode(chain::Chain, xz, z::AA, xt)
    for f in chain
        xz, z = decode(f, xz, z, xt)
    end
    return xz, z
end

"""
    abstract type Aggregator
"""
abstract type Aggregator end

function encode(agg::Aggregator, xc::AA, yc::AA, xt::AA; kws...)
    xz = encoding_locations(agg, xc, xt; kws...)
    return encode(agg, xc, yc, xt, xz; kws...)
end

function encode(agg::Aggregator, xc::AA, yc::AA, xt::AA, xz::AA; kws...)
    size(xc, 1) == 0 && (xc = yc = nothing)
    return xz, AggregateEncoding([
        second(encode(encoder, xc, yc, xz; kws...)) for encoder in agg.encoders
    ])
end

"""
    reencode_stochastic(
        agg::Aggregator,
        agg_encoding::AggregateEncoding,
        xc::AA,
        yc::AA,
        xt::AA,
        xz::AA;
        kws...
    )

In an existing aggregate encoding `agg_encoding`, eeencode the encodings that are not
`Dirac`s for a new context and target set.

# Arguments
- `agg::Aggregator`: Aggregated encoders that produced the encoding.
- `agg_encoding::AggregatedEncoding`: Aggregate encoding.
- `xc::AA`: Locations of context set of shape `(n, dims, batch)`.
- `yc::AA`: Observed values of context set of shape `(n, channels, batch)`.
- `xt::AA`: Locations of target set of shape `(m, dims, batch)`.
- `yt::AA`: Observed values of target set of shape `(m, channels, batch)`.

# Fields
- `kws...`: Further keywords to pass on.

# Returns
- Updated encoding.
"""
function reencode_stochastic(
    agg::Aggregator,
    agg_encoding::AggregateEncoding,
    xc::AA,
    yc::AA,
    xt::AA,
    xz::AA;
    kws...
)
    return xz, AggregateEncoding([
        second(reencode_stochastic(encoder, encoding, xc, yc, xz; kws...))
        for (encoder, encoding) in zip(agg.encoders, agg_encoding.encodings)
    ])
end

reencode_stochastic(encoder, encoding::Dirac, xz::AA, z::AA, x::AA; kws...) = x, encoding
reencode_stochastic(encoder, encoding, xc, yc, xz; kws...) =
    encode(encoder, xc, yc, xz; kws...)

"""
    struct TargetAggregator <: Aggregator

Aggregation of encoders that encode at the target set locations.

# Fields
- `encoders::Vector`: Encoders.
"""
struct TargetAggregator <: Aggregator
    encoders::Vector
end

Flux.children(agg::TargetAggregator) = agg.encoders
Flux.mapchildren(f, agg::TargetAggregator) = TargetAggregator(f.(agg.encoders)...)

TargetAggregator(encoders...) = TargetAggregator(collect(encoders))
encoding_locations(agg::TargetAggregator, xc, xt::AA; kws...) = xt

"""
    struct TargetAggregator <: Aggregator

Aggregation of encoders that encode at a discretisation determined by the context and
target inputs.

# Fields
- `disc::Discretisation`: Discretisation.
- `encoders::Vector`: Encoders.
"""
struct FunctionalAggregator <: Aggregator
    disc::Discretisation
    encoders::Vector
end

Flux.children(agg::FunctionalAggregator) = agg.encoders
Flux.mapchildren(f, agg::FunctionalAggregator) =
    FunctionalAggregator(agg.disc, f.(agg.encoders)...)

FunctionalAggregator(disc::Discretisation, encoders...) =
    FunctionalAggregator(disc::Discretisation, collect(encoders))
encoding_locations(agg::FunctionalAggregator, xc::AA, xt::AA; kws...) =
    agg.disc(xc, xt; kws...)
