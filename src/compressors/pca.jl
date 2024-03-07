import MultivariateStats

# TODO: ask Mykel if there's a better way to do this
mutable struct PCA <: Compressor
    n_components::Integer
    M
end

PCA(n_components::Integer) = PCA(n_components, nothing)

function fit!(compressor::PCA, beliefs)
    compressor.M = MultivariateStats.fit(MultivariateStats.PCA, beliefs; maxoutdim=compressor.n_components)
end

# compress(compressor::PCA, beliefs) = MultivariateStats.predict(compressor.M, beliefs)
# decompress(compressor::PCA, compressed) = MultivariateStats.reconstruct(compressor.M, compressed)

function compress(compressor::PCA, beliefs)
    @infiltrate
    return MultivariateStats.predict(compressor.M, beliefs)
end

function decompress(compressor::PCA, compressed)
    @infiltrate
    return MultivariateStats.reconstruct(compressor.M, compressed)
end