abstract type Compressor end

"""
    fit!(compressor::Compressor, beliefs; kwargs...)

Fit the compressor to beliefs.
"""
function fit! end