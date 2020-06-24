module ConvCNPs

using Distributions
using Flux
using LinearAlgebra
using NNlib
using Printf
using Random
using Statistics
using StatsBase

if Flux.has_cuarrays()
    include("gpu.jl")

    using CuArrays
    randn_gpu = CuArrays.randn
    zeros_gpu = CuArrays.zeros
    ones_gpu = CuArrays.ones
else
    const CuOrVector = Vector
    const CuOrMatrix = Matrix
    const CuOrArray = Array

    randn_gpu = randn
    zeros_gpu = zeros
    ones_gpu = ones
end

const AV = AbstractVector
const AM = AbstractMatrix
const AA = AbstractArray

const MaybeAV = Union{Nothing, AbstractVector}
const MaybeAM = Union{Nothing, AbstractMatrix}
const MaybeAA = Union{Nothing, AbstractArray}

include("util.jl")
include("data.jl")
include("conv.jl")
include("discretisation.jl")
include("parallel.jl")
include("nn.jl")
include("distribution.jl")

include("model/setconv.jl")
include("model/attention.jl")
include("model/coding.jl")
include("model/noise.jl")
include("model/model.jl")
include("model/convcnp.jl")
include("model/convnp.jl")
include("model/np.jl")
include("model/anp.jl")
# include("model/correlatedconvcnp.jl")

include("experiment/experiment.jl")

end
