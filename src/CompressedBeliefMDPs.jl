module CompressedBeliefMDPs

export
    Compressor,
    fit!,
    compress,
    decompress
include("compressor.jl")

export
    Sampler,
    BaseSampler,
    DiscreteRandomSampler,
    sample
include("sampling.jl")

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
