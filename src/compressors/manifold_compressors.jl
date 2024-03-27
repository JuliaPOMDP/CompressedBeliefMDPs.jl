"""
Wrappers for MultivariateStats.jl. See https://juliastats.org/MultivariateStats.jl/stable/.
"""

using ManifoldLearning


mutable struct ManifoldCompressor{T} <: Compressor
    const maxoutdim::Integer
    M
    kwargs
end

ManifoldCompressor(maxoutdim::Integer, T, kwargs) = ManifoldCompressor{T}(maxoutdim, missing, kwargs)

function (c::ManifoldCompressor)(beliefs)
    if ismissing(c.M)
        @warn "compressor.M is missing"
    end
    return ndims(beliefs) == 2 ? ManifoldLearning.predict(c.M, beliefs')' : vec(ManifoldLearning.predict(c.M, beliefs))
end

function fit!(c::ManifoldCompressor{T}, beliefs) where T
    c.M = ManifoldLearning.fit(T, beliefs'; maxoutdim=c.maxoutdim, c.kwargs...)
end

### ManifoldLearning.jl Wrappers ###
IsomapCompressor(maxoutdim::Integer; kwargs...) = ManifoldCompressor(maxoutdim, Isomap, kwargs)

### Discontinued ###

# function (c::ManifoldCompressor{T})(beliefs) where T
#     M = ManifoldLearning.fit(T, beliefs'; maxoutdim=c.maxoutdim, c.kwargs...)
#     R = ManifoldLearning.predict(M)
#     # R = ndims(beliefs) == 2 ? ManifoldLearning.predict(M, beliefs')' : vec(ManifoldLearning.predict(M, beliefs))
#     return R
# end

# function fit!(compressor::ManifoldCompressor{T}, beliefs; kwargs...) where T
#     compressor.kwargs = kwargs
# end

# LLECompressor(maxoutdim::Integer) = ManifoldCompressor(maxoutdim, LLE)
# HLLECompressor(maxoutdim::Integer) = ManifoldCompressor(maxoutdim, HLLE)
# LEMCompressor(maxoutdim::Integer) = ManifoldCompressor(maxoutdim, LEM)
# LTSACompressor(maxoutdim::Integer) = ManifoldCompressor(maxoutdim, LTSA)
# DiffMapCompressor(maxoutdim::Integer) = ManifoldCompressor(maxoutdim, DiffMap)


