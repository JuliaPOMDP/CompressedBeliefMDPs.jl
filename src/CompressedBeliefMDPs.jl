module CompressedBeliefMDPs

# Packages from JuliaPOMDPs
using POMDPs, POMDPTools, POMDPModels
using ParticleFilters
using LocalApproximationValueIteration
using LocalFunctionApproximation

# Other 3P Packages
using Distributions
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
    ### ManifoldLearning wrappers ###
    ManifoldCompressor,
    IsomapCompressor,
    ### Flux compressors ###
    AutoencoderCompressor,
    VAECompressor
include("compressors/compressor.jl")
include("compressors/mvs_compressors.jl")
include("compressors/manifold_compressors.jl")
include("compressors/autoencoders.jl")
include("compressors/vae.jl")

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

export
    CircularCorridorPOMDP,
    CircularCorridorState
include("envs/circular.jl")

end # module CompressedBeliefMDPs
