using ConvCNPs
using FiniteDifferences
using Flux
using Flux.Tracker
using LinearAlgebra
using Test

@testset "ConvCNPs.jl" begin
    include("util.jl")
    include("discretisation.jl")
    include("setconv.jl")
    include("attention.jl")
    include("data.jl")
end
