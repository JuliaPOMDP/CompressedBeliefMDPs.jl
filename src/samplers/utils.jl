# TODO: replace w/ take (get GPT to simplify this loop)
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
    policy::Policy
    updater::Updater
end

sample(sampler::BaseSampler, pomdp::POMDP; n_samples::Integer=100) = sample(pomdp, n_samples; sampler.policy, sampler.updater)
DiscreteSampler(pomdp::POMDP, policy::Policy) = BaseSampler(policy, DiscreteUpdater(pomdp))

function DiscreteEpsGreedySampler(pomdp::POMDP, eps; rng::AbstractRNG=Random.GLOBAL_RNG)
    policy = EpsGreedyPolicy(pomdp, eps; rng=rng)
    return DiscreteSampler(pomdp, policy)
end

# TODO: figure out a better default schedule
# TODO: replace this w/ custom eps greedy policy
function DiscreteSoftmaxSampler(pomdp::POMDP, temperature; rng::AbstractRNG=Random.GLOBAL_RNG)
    policy = SoftmaxPolicy(pomdp, temperature; rng=rng)
    return DiscreteSampler(pomdp, policy)
end

function DiscreteRandomSampler(pomdp::POMDP; rng::AbstractRNG=Random.GLOBAL_RNG)
    updater = DiscreteUpdater(pomdp)
    policy = RandomPolicy(pomdp; rng=rng, updater=updater)
    return BaseSampler(policy, updater)
end