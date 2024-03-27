struct PolicySampler <: Sampler
    policy::Policy
    updater::Updater
    n::Integer
    rng::AbstractRNG
end

function PolicySampler(
    pomdp::POMDP;
    policy::Policy=RandomPolicy(pomdp), 
    updater::Updater=DiscreteUpdater(pomdp), 
    n=10,
    rng::AbstractRNG=Random.GLOBAL_RNG
)
    PolicySampler(policy, updater, n, rng)
end

function (s::PolicySampler)(pomdp::POMDP)
    mdp = GenerativeBeliefMDP(pomdp, s.updater)
    iter = stepthrough(mdp, s.policy, "s", rng=s.rng; max_steps=s.n)
    B = collect(Iterators.take(Iterators.cycle(iter), s.n))
end

struct ExplorationPolicySampler <: Sampler
    explorer::ExplorationPolicy
    on_policy::Policy
    updater::Updater
    n::Integer
    rng::AbstractRNG
end

function ExplorationPolicySampler(pomdp::POMDP; 
    rng::AbstractRNG=Random.GLOBAL_RNG,
    explorer::ExplorationPolicy=EpsGreedyPolicy(pomdp, 0.1; rng=rng),
    on_policy=RandomPolicy(pomdp),
    updater::Updater=DiscreteUpdater(pomdp), 
    n=10, 
)
    ExplorationPolicySampler(explorer, on_policy, updater, n, rng)
end

function (s::ExplorationPolicySampler)(pomdp::POMDP)
    B = []
    mdp = GenerativeBeliefMDP(pomdp, s.updater)
    while true
        b = initialstate(mdp).val
        for k in 1:s.n
            if length(B) == s.n
                return unique!(B)
            end

            if isterminal(mdp, b)
                break
            end
            a = action(s.explorer, s.on_policy, k, b)
            b = @gen(:sp)(mdp, b, a, s.rng)
            push!(B, b)
        end
    end
    return B
end

