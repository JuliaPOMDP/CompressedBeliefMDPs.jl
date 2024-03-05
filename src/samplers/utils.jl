using POMDPs: POMDP, Policy, Updater
using POMDPPolicies
using POMDPTools: RandomPolicy, DiscreteUpdater
using POMDPSimulators: stepthrough


function sample(pomdp::POMDP, n_samples::Integer; policy::Policy=RandomPolicy(pomdp), updater::Updater=DiscreteUpdater(pomdp))
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


struct BaseSampler <: Sampler
    n_samples::Integer
    policy::Policy
    updater::Updater
end

sample(sampler::BaseSampler, pomdp::POMDP) = sample(pomdp, sampler.n_samples, sampler.policy, sampler.updater)

DiscreteSampler(pomdp::POMDP, n_samples::Integer, policy::Policy) = BaseSampler(n_samples, policy, DiscreteUpdater(pomdp))

function DiscreteEpsGreedySampler(pomdp::POMDP, n_samples::Integer; eps::Function=k->0.05*0.9^(k/10), rng::AbstractRNG)
    policy = EpsGreedyPolicy(pomdp, eps; actions=POMDPs.actions(pomdp), rng=rng)
    return DiscreteSampler(pomdp, n_samples, policy)
end

function DiscreteSoftmaxSampler(pomdp::POMDP, n_samples::Integer; temperature::Function=k->0.05*0.9^(k/10), rng::AbstractRNG)
    policy = SoftmaxPolicy(pomdp, temperature; actions=POMDPs.actions(pomdp), rng=rng)
    return DiscreteSampler(pomdp, n_samples, policy)
end

function DiscreteRandomSampler(pomdp::POMDP, n_samples::Integer; rng::AbstractRNG)
    updater = DiscreteUpdater(pomdp)
    policy = RandomPolicy(pomdp; rng=rng, updater=updater)
    return BaseSampler(n_samples, policy, updater)
end