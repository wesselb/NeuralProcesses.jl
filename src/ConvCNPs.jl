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
include("nn.jl")
include("setconv.jl")
include("attention.jl")
include("model/cnp/convcnp.jl")
include("model/cnp/correlatedconvcnp.jl")
include("model/np/np.jl")
include("model/np/convnp.jl")
include("model/np/anp.jl")

include("experiment/experiment.jl")

end
