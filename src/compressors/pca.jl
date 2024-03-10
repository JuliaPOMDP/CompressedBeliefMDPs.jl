import MultivariateStats

# TODO: ask Mykel if way to make this immutable
mutable struct PCA <: Compressor
    n_components::Integer
    M
end

PCA(n_components::Integer) = PCA(n_components, nothing)

function fit!(compressor::PCA, beliefs)
    compressor.M = MultivariateStats.fit(MultivariateStats.PCA, beliefs'; maxoutdim=compressor.n_components)
end

function compress(compressor::PCA, beliefs) 
    if ndims(beliefs) == 2
        return MultivariateStats.predict(compressor.M, beliefs')'
    else
        return MultivariateStats.predict(compressor.M, beliefs)
    end
end

decompress(compressor::PCA, compressed) = MultivariateStats.reconstruct(compressor.M, compressed)