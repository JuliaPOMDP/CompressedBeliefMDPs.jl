module CompressedBeliefMDPs


using Infiltrator

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
include("compressed_belief_mdp.jl")

end # module CompressedBeliefMDPs
