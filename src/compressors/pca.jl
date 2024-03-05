using POMDPs
using POMDPTools: DiscreteBelief
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

compress(compressor::PCA, beliefs) = MultivariateStats.predict(compressor.M, beliefs)
decompress(compressor::PCA, compressed) = MultivariateStats.reconstruct(compressor.M, compressed)

# TODO: replace this w/ a generic normalization pipeline when there are negative values
decode(m::POMDP, c::PCA, b̃) = DiscreteBelief(m, vec(normalize(abs.(decompress(c, b̃')), 1)))
