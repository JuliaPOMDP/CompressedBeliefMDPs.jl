module CompressedBeliefMDPs

using Infiltrator

using POMDPs
using POMDPTools
using LocalApproximationValueIteration
using LocalFunctionApproximation

import Lazy: @forward
using Bijections
using NearestNeighbors
using StaticArrays

using LinearAlgebra
using Random


export
    # Compressor interface
    Compressor,
    fit!,
    compress,
    decompress,
    # MultivariateStats Compressors
    MultivariateStatsCompressor,
    PCACompressor,
    KernelPCACompressor,
    PPCACompressor
    # FactorAnalysisCompressor  # TODO: debug
include("compressors/compressor.jl")
include("compressors/mv_stats.jl")

export
    sample
include("sampler.jl")

export
    CompressedBeliefMDP,
include("cbmdp.jl")

export
    CompressedBeliefSolver,
    CompressedBeliefPolicy,
    solve
include("solver.jl")

end # module CompressedBeliefMDPs
