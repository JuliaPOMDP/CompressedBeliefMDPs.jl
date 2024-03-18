module CompressedBeliefMDPs

using Infiltrator

using POMDPs
using POMDPTools

import Lazy: @forward

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
    Sampler,
    sample,
    # DiscreteEpsGreedySampler,
    # DiscreteSoftmaxSampler,
    DiscreteRandomSampler
include("samplers/sampler.jl")
include("samplers/utils.jl")

export
    CompressedBeliefMDP,
    CompressedBeliefMDPState
include("cbmdp.jl")

export
    CompressedSolver,
    CompressedSolverPolicy
    solve
include("solver.jl")

end # module CompressedBeliefMDPs
