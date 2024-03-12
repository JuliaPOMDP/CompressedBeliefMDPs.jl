abstract type Sampler end

"""
    sample(sampler::Sampler, pomdp::POMDP; n_samples::Integer=100)

Return a matrix of beliefs sampled from pomdp.
"""
function sample end
