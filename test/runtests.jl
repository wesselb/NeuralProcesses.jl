using ConvCNPs
using Test

@testset "ConvCNPs.jl" begin
    include("util.jl")
    include("setconv.jl")
    include("discretisation.jl")
    include("data.jl")
end
