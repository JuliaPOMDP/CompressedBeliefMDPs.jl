using CompressedBeliefMDPs
using Test

using POMDPs, POMDPModels, POMDPTools
using MCTS

@testset "CompressedBeliefMDPs.jl" begin
    include("compressor_tests.jl")
    include("solver_tests.jl")
end
