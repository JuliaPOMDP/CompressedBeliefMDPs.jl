abstract type Sampler end

"""
    sample(sampler::Sampler, pomdp::POMDP)

Return a matrix of beliefs sampled from pomdp.
"""
function sample end