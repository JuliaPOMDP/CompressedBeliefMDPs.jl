module CompressedBeliefMDPs

using Infiltrator

using POMDPs, POMDPTools, POMDPModels
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
    ### Compressor Interface ###
    Compressor,
    fit!,
    ### MultivariateStats wrappers ###
    MVSCompressor,
    PCACompressor,
    KernelPCACompressor,
    PPCACompressor,
    FactorAnalysisCompressor,
    # MDSCompressor,
    ### ManifoldLearning wrappers ###
    ManifoldCompressor,
    IsomapCompressor,
    ### Flux compressors ###
    AutoencoderCompressor
include("compressors/compressor.jl")
include("compressors/mvs_compressors.jl")
include("compressors/manifold_compressors.jl")
include("compressors/autoencoders.jl")

export
    Sampler,
    sample,
    BeliefExpansionSampler,
    PolicySampler,
    ExplorationPolicySampler
include("samplers/samplers.jl")
include("samplers/expansion.jl")
include("samplers/rollout.jl")

export
    CompressedBeliefMDP
include("cbmdp.jl")

export
    CompressedBeliefSolver,
    CompressedBeliefPolicy,
    solve
include("solver.jl")

end # module CompressedBeliefMDPs
