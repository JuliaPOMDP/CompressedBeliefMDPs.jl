"""
Abstract type for an object that defines how the belief should be compressed.
"""
abstract type Compressor end

"""
    fit!(compressor::Compressor, beliefs)

Fit the compressor to beliefs.
"""
function fit! end