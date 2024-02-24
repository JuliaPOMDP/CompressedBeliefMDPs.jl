"""
Base type for an MDP/POMDP belief compression
"""
abstract type Compressor end

"""
    fit!(compressor::Compressor, beliefs::AbstractArray)

Fit the compressor to beliefs.
"""
function fit! end

# TODO: consider something like this: abstract type Pointy{T<:Real} end; see (https://docs.julialang.org/en/v1/manual/types/)
"""
    compress(compressor::Compressor, beliefs::AbstractArray)

Compress the sampled beliefs using method associated with compressor, and returns a compressed representation.
"""
function compress end

"""
    decompress(compressor::Compressor, compressed::AbstractArray)

Decompress the compressed beliefs using method associated with compressor, and returns the reconstructed beliefs.
"""
function decompress end