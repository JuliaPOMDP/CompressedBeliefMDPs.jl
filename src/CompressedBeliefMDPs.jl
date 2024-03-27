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
    IsomapCompressor
    # LLECompressor,
    # HLLECompressor,
    # LEMCompressor,
    # LTSACompressor,
    # DiffMapCompressor
include("compressors/compressor.jl")
include("compressors/mvs_compressors.jl")
include("compressors/manifold_compressors.jl")

export
    sample,
    exploratory_belief_expansion
include("samplers/policy_simulation.jl")
include("samplers/belief_expansion.jl")

export
    CompressedBeliefMDP
include("cbmdp.jl")

export
    CompressedBeliefSolver,
    CompressedBeliefPolicy,
    solve
include("solver.jl")

end # module CompressedBeliefMDPs
