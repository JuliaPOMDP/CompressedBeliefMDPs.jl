# TODO: add RNG support for seeding

function sample(pomdp::POMDP, explorer::Policy, updater::Updater, n::Integer; rng::AbstractRNG=Random.GLOBAL_RNG)
    mdp = GenerativeBeliefMDP(pomdp, updater)
    iter = stepthrough(mdp, explorer, "s"; max_steps=n)
    B = collect(Iterators.take(Iterators.cycle(iter), n))
    return unique!(B)
end

function sample(pomdp::POMDP, explorer::ExplorationPolicy, updater::Updater, n::Integer; rng::AbstractRNG=Random.GLOBAL_RNG)
    samples = []
    mdp = GenerativeBeliefMDP(pomdp, updater)
    on_policy = RandomPolicy(mdp)
    while true
        b = initialstate(mdp).val
        for k in 1:n
            if length(samples) == n
                return unique!(samples)
            end

            if isterminal(mdp, b)
                break
            end
            a = action(explorer, on_policy, k, b)
            b = @gen(:sp)(mdp, b, a, rng)
            push!(samples, b)
        end
    end
end