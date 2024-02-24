using POMDPs: POMDP, Policy, Updater
using POMDPTools: RandomPolicy, DiscreteUpdater
using POMDPSimulators: stepthrough


abstract type Sampler end

"""
    sample(sampler::Sampler, pomdp::POMDP)

Return a matrix of beliefs sampled from pomdp.
"""
function sample end

# TODO: try doing policy=RandomPolicy(pomdp)
function sample(pomdp::POMDP, n_samples::Int=100, policy::Policy=RandomPolicy(pomdp), updater::Updater=DiscreteUpdater(pomdp))
    B = []
    while true
        for dist in stepthrough(pomdp, policy, updater, "b", max_steps=n_samples)
            push!(B, dist.b)
            if length(B) == n_samples
                return hcat(B...)'  # convert to n_samples x n_states matrix
            end
        end
    end
end

# TODO: can I use template type T to construct policy and updater? play around w/ this
struct BaseSampler <: Sampler
    n_samples::Int
    policy::Policy
    updater::Updater
end

sample(sampler::BaseSampler, pomdp::POMDP) = sample(pomdp, sampler.n_samples, sampler.policy, sampler.updater)
DiscreteRandomSampler(pomdp::POMDP; n_samples::Int=100) = BaseSampler(n_samples, RandomPolicy(pomdp), DiscreteUpdater(pomdp))