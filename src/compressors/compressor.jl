"""
Base type for an MDP/POMDP belief compression.
"""
abstract type Compressor end


"""
    fit!(compressor::Compressor, beliefs)

Fit the compressor to beliefs.
"""
function fit! end


"""
    compress(compressor::Compressor, beliefs)

Compress the sampled beliefs using method associated with compressor, and returns a compressed representation.
"""
function compress end


"""
    decompress(compressor::Compressor, compressed)

Decompress the compressed beliefs using method associated with compressor, and returns the reconstructed beliefs.
"""
function decompress end