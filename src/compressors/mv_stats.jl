using MultivariateStats

mutable struct MultivariateStatsCompressor{T<:MultivariateStats.AbstractDimensionalityReduction} <: Compressor
    const maxoutdim::Integer
    M  # TODO: check if this is Julian (how to replace unde)
end

function fit!(compressor::MultivariateStatsCompressor{T}, beliefs) where T<:MultivariateStats.AbstractDimensionalityReduction
    compressor.M = MultivariateStats.fit(T, beliefs'; maxoutdim=compressor.maxoutdim)
end

# TODO: is there a way to solve this w/ multiple dispatch? clean up
function compress(compressor::MultivariateStatsCompressor, beliefs)
    # TODO: is there better way to do this?
    return ndims(beliefs) == 2 ? predict(compressor.M, beliefs')' : vec(predict(compressor.M, beliefs))
end

decompress(compressor::MultivariateStatsCompressor, compressed) = MultivariateStats.reconstruct(compressor.M, compressed)

MultivariateStatsCompressor(maxoutdim::Integer, T) = MultivariateStatsCompressor{T}(maxoutdim, nothing)

# PCA Compressors
PCACompressor(maxoutdim::Integer) = MultivariateStatsCompressor(maxoutdim, PCA)
KernelPCACompressor(maxoutdim::Integer) = MultivariateStatsCompressor(maxoutdim, KernelPCA)
PPCACompressor(maxoutdim::Integer) = MultivariateStatsCompressor(maxoutdim, PPCA)

# TODO: debug this
function fit!(compressor::MultivariateStatsCompressor{KernelPCA}, beliefs)
    compressor.M = MultivariateStats.fit(KernelPCA, beliefs'; maxoutdim=compressor.maxoutdim, inverse=true)
end

# Factor Analysis Compressor
FactorAnalysisCompressor(maxoutdim::Integer) = MultivariateStatsCompressor(maxoutdim, FactorAnalysis)

