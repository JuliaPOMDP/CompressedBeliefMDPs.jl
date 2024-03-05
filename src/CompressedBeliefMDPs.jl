module CompressedBeliefMDPs

using Infiltrator


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
    DiscreteRandomSampler,
    sample
include("samplers/sampler.jl")
include("samplers/utils.jl")

export
    CompressedSolver,
    solve
include("solver.jl")

export
    CompressedBeliefMDP,
    encode,
    decode
include("compressed_bmdp.jl")

end # module CompressedBeliefMDPs
