"""
Wrappers for MultivariateStats.jl. See https://juliastats.org/MultivariateStats.jl/stable/.
"""

using MultivariateStats


mutable struct MVSCompressor{T<:MultivariateStats.AbstractDimensionalityReduction} <: Compressor
    const maxoutdim::Integer
    M
end

(c::MVSCompressor)(beliefs) = ndims(beliefs) == 2 ? MultivariateStats.predict(c.M, beliefs')' : vec(MultivariateStats.predict(c.M, beliefs))

function fit!(compressor::MVSCompressor{T}, beliefs) where T<:MultivariateStats.AbstractDimensionalityReduction
    compressor.M = MultivariateStats.fit(T, beliefs'; maxoutdim=compressor.maxoutdim)
end

MVSCompressor(maxoutdim::Integer, T) = MVSCompressor{T}(maxoutdim, nothing)

# PCA Compressors
PCACompressor(maxoutdim::Integer) = MVSCompressor(maxoutdim, PCA)
KernelPCACompressor(maxoutdim::Integer) = MVSCompressor(maxoutdim, KernelPCA)
PPCACompressor(maxoutdim::Integer) = MVSCompressor(maxoutdim, PPCA)

# Factor Analysis Compressor
FactorAnalysisCompressor(maxoutdim::Integer) = MVSCompressor(maxoutdim, FactorAnalysis)

# Multidimensional Scaling
MDSCompressor(maxoutdim::Integer) = MVSCompressor(maxoutdim, MDS)