module CompressedBeliefMDPs

export
    Compressor,
    fit!,
    compress,
    decompress
include("compressors/compressor.jl")

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
