"""
Wrappers for MultivariateStats.jl. See https://juliastats.org/MultivariateStats.jl/stable/.
"""

using MultivariateStats
using MultivariateStats: predict


mutable struct MVSCompressor{T} <: Compressor
    const maxoutdim::Integer
    M
    kwargs
end

MVSCompressor(maxoutdim::Integer, T, kwargs) = MVSCompressor{T}(maxoutdim, missing, kwargs)

function (c::MVSCompressor)(beliefs)
    if ismissing(c.M)
        @warn "compressor.M is missing"
    end
    return ndims(beliefs) == 2 ? predict(c.M, beliefs')' : vec(predict(c.M, beliefs))
end

function fit!(c::MVSCompressor{T}, beliefs) where T
    # c.M = isempty(kwargs) ? fit(T, beliefs'; maxoutdim=c.maxoutdim) : fit(T, beliefs'; maxoutdim=c.maxoutdim, v.kwargs...)
    c.M = fit(T, beliefs'; maxoutdim=c.maxoutdim, c.kwargs...)
end

### MultivariateStats.jl Wrappers ###
# PCA Compressors
PCACompressor(maxoutdim::Integer; kwargs...) = MVSCompressor(maxoutdim, PCA, kwargs)
KernelPCACompressor(maxoutdim::Integer; kwargs...) = MVSCompressor(maxoutdim, KernelPCA, kwargs)
PPCACompressor(maxoutdim::Integer; kwargs...) = MVSCompressor(maxoutdim, PPCA, kwargs)

# Factor Analysis Compressor
FactorAnalysisCompressor(maxoutdim::Integer; kwargs...) = MVSCompressor(maxoutdim, FactorAnalysis, kwargs)

# Multidimensional Scaling
# MDSCompressor(maxoutdim::Integer) = MVSCompressor(maxoutdim, MDS)

