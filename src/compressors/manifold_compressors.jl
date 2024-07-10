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
    if ndims(beliefs) == 2
        result = ManifoldLearning.predict(c.M, beliefs')'
    else
        result = vec(ManifoldLearning.predict(c.M, beliefs))
    end
    return result
end

function fit!(c::ManifoldCompressor{T}, beliefs) where T
    c.M = ManifoldLearning.fit(T, beliefs'; maxoutdim=c.maxoutdim, c.kwargs...)
end

### ManifoldLearning.jl Wrappers ###

"""Wrapper for [`ManifoldLearning.Isomap`](https://wildart.github.io/ManifoldLearning.jl/stable/isomap/#Isomap)."""
IsomapCompressor(maxoutdim::Integer; kwargs...) = ManifoldCompressor(maxoutdim, Isomap, kwargs)