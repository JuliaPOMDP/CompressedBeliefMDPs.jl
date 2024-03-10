module CompressedBeliefMDPs

using Infiltrator

using POMDPs
using POMDPTools

using LinearAlgebra
using Random


export
    Compressor,
    fit!,
    compress,
    decompress,
    PCA
include("compressors/compressor.jl")
include("compressors/pca.jl")

export
    Sampler,
    sample,
    DiscreteEpsGreedySampler,
    DiscreteSoftmaxSampler,
    DiscreteRandomSampler
include("samplers/sampler.jl")
include("samplers/utils.jl")

export
    CompressedBeliefMDP
include("cbmdp.jl")

export
    CompressedSolver,
    solve
include("solver.jl")

end # module CompressedBeliefMDPs
