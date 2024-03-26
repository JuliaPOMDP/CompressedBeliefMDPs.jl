module CompressedBeliefMDPs

using Infiltrator

using POMDPs
using POMDPTools
using ParticleFilters
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
    # StatsAPI Compressors
    StatsCompressor,
    ## MultivariateStats wrappers
    PCACompressor,
    KernelPCACompressor,
    PPCACompressor,
    FactorAnalysisCompressor,
    MDSCompressor,
    ## ManifoldLearning wrappers
    IsomapCompressor,
    LLECompressor,
    HLLECompressor,
    LEMCompressor,
    LTSACompressor,
    DiffMapCompressor
include("compressors/compressor.jl")
include("compressors/stats_compressors.jl")

export
    sample
include("sampler.jl")

export
    CompressedBeliefMDP
include("cbmdp.jl")

export
    CompressedBeliefSolver,
    CompressedBeliefPolicy,
    solve
include("solver.jl")

end # module CompressedBeliefMDPs
