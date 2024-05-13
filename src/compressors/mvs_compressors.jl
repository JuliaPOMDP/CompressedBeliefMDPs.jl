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
    c.M = fit(T, beliefs'; maxoutdim=c.maxoutdim, c.kwargs...)
end

### MultivariateStats.jl Wrappers ###
# PCA Compressors

"""Wrapper for [`MultivariateStats.PCA`](https://juliastats.org/MultivariateStats.jl/stable/pca/#Linear-Principal-Component-Analysis)."""
function PCACompressor(maxoutdim::Integer; kwargs...)
    compressor = MVSCompressor(maxoutdim, PCA, kwargs)
    return compressor
end


"""Wrapper for [`MultivariateStats.KernelPCA`](https://juliastats.org/MultivariateStats.jl/stable/pca/#Kernel-Principal-Component-Analysis)."""
function KernelPCACompressor(maxoutdim::Integer; kwargs...)
    compressor = MVSCompressor(maxoutdim, KernelPCA, kwargs)
    return compressor
end


"""Wrapper for [`MultivariateStats.PPCA`](https://juliastats.org/MultivariateStats.jl/stable/pca/#Probabilistic-Principal-Component-Analysis)."""
function PPCACompressor(maxoutdim::Integer; kwargs...)
    compressor = MVSCompressor(maxoutdim, PPCA, kwargs)
    return compressor
end


# Factor Analysis Compressor
"""Wrapper for [`MultivariateStats.FactorAnalysis`](https://juliastats.org/MultivariateStats.jl/stable/fa/)"""
function FactorAnalysisCompressor(maxoutdim::Integer; kwargs...)
    compressor = MVSCompressor(maxoutdim, FactorAnalysis, kwargs)
    return compressor
end

# Multidimensional Scaling
# MDSCompressor(maxoutdim::Integer) = MVSCompressor(maxoutdim, MDS)

