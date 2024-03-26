"""
Wrappers for MultivariateStats.jl. See https://juliastats.org/MultivariateStats.jl/stable/.
Wrapper for ManifoldLearning.jl. See https://wildart.github.io/ManifoldLearning.jl/stable/.
"""

using MultivariateStats
using ManifoldLearning
import StatsAPI


mutable struct StatsCompressor{T} <: Compressor
    const maxoutdim::Integer
    M
end

StatsCompressor(maxoutdim::Integer, T) = StatsCompressor{T}(maxoutdim, missing)

function (c::StatsCompressor)(beliefs)
    # @infiltrate
    return ndims(beliefs) == 2 ? StatsAPI.predict(c.M, beliefs')' : vec(StatsAPI.predict(c.M, beliefs))
end

function fit!(compressor::StatsCompressor{T}, beliefs; kwargs...) where T
    compressor.M = StatsAPI.fit(T, beliefs'; maxoutdim=compressor.maxoutdim, kwargs...)
end

### MultivariateStats.jl Wrappers ###
# PCA Compressors
PCACompressor(maxoutdim::Integer) = StatsCompressor(maxoutdim, PCA)
KernelPCACompressor(maxoutdim::Integer) = StatsCompressor(maxoutdim, KernelPCA)
PPCACompressor(maxoutdim::Integer) = StatsCompressor(maxoutdim, PPCA)

# Factor Analysis Compressor
FactorAnalysisCompressor(maxoutdim::Integer) = StatsCompressor(maxoutdim, FactorAnalysis)

# Multidimensional Scaling
MDSCompressor(maxoutdim::Integer) = StatsCompressor(maxoutdim, MDS)

### ManifoldLearning.jl Wrappers ###
IsomapCompressor(maxoutdim::Integer) = StatsCompressor(maxoutdim, Isomap)
LLECompressor(maxoutdim::Integer) = StatsCompressor(maxoutdim, LLE)
HLLECompressor(maxoutdim::Integer) = StatsCompressor(maxoutdim, HLLE)
LEMCompressor(maxoutdim::Integer) = StatsCompressor(maxoutdim, LEM)
LTSACompressor(maxoutdim::Integer) = StatsCompressor(maxoutdim, LTSA)
DiffMapCompressor(maxoutdim::Integer) = StatsCompressor(maxoutdim, DiffMap)

