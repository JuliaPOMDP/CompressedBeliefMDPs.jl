"""
Wrappers for MultivariateStats.jl. See https://juliastats.org/MultivariateStats.jl/stable/.
"""

using MultivariateStats
using MultivariateStats: predict


mutable struct MVSCompressor{T} <: Compressor
    const maxoutdim::Integer
    M
end

MVSCompressor(maxoutdim::Integer, T) = MVSCompressor{T}(maxoutdim, missing)

function (c::MVSCompressor)(beliefs)
    return ndims(beliefs) == 2 ? predict(c.M, beliefs')' : vec(predict(c.M, beliefs))
end

function fit!(compressor::MVSCompressor{T}, beliefs; kwargs...) where T
    compressor.M = fit(T, beliefs'; maxoutdim=compressor.maxoutdim, kwargs...)
end

### MultivariateStats.jl Wrappers ###
# PCA Compressors
PCACompressor(maxoutdim::Integer) = MVSCompressor(maxoutdim, PCA)
KernelPCACompressor(maxoutdim::Integer) = MVSCompressor(maxoutdim, KernelPCA)
PPCACompressor(maxoutdim::Integer) = MVSCompressor(maxoutdim, PPCA)

# Factor Analysis Compressor
FactorAnalysisCompressor(maxoutdim::Integer) = MVSCompressor(maxoutdim, FactorAnalysis)

# Multidimensional Scaling
# MDSCompressor(maxoutdim::Integer) = MVSCompressor(maxoutdim, MDS)

