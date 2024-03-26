function sample(pomdp::POMDP, policy::Policy, updater::Updater, n::Integer)
    mdp = GenerativeBeliefMDP(pomdp, updater)
    iter = stepthrough(mdp, policy, "s"; max_steps=n)
    B = collect(Iterators.take(Iterators.cycle(iter), n))
    return unique!(B)
end

function sample(pomdp::POMDP, policy::ExplorationPolicy, updater::Updater, n::Integer)
    # TODO:
end


"""
Adapted from algorithm 21.13 from AFDM
"""