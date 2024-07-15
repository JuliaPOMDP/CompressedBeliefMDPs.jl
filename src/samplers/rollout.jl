"""
    PolicySampler

Samples belief states by rolling out a `Policy`.

## Fields
- `policy::Policy`: The policy used for decision making.
- `updater::Updater`: The updater used for updating beliefs.
- `n::Integer`: The maximum number of simulated steps.
- `rng::AbstractRNG`: The random number generator used for sampling.
- `verbose::Bool`: Whether to use a progress bar while sampling.


## Constructors
    PolicySampler(pomdp::POMDP; policy::Policy=RandomPolicy(pomdp), 
    updater::Updater=DiscreteUpdater(pomdp), n::Integer=10, 
    rng::AbstractRNG=Random.GLOBAL_RNG)

## Methods
    (s::PolicySampler)(pomdp::POMDP)
    
Returns a vector of _unique_ belief states.

# Example
```julia-repl
julia> pomdp = TigerPOMDP();
julia> sampler = PolicySampler(pomdp; n=3); 
julia> 2-element Vector{Any}:
DiscreteBelief{TigerPOMDP, Bool}(TigerPOMDP(-1.0, -100.0, 10.0, 0.85, 0.95), Bool[0, 1], [0.5, 0.5])
DiscreteBelief{TigerPOMDP, Bool}(TigerPOMDP(-1.0, -100.0, 10.0, 0.85, 0.95), Bool[0, 1], [0.15000000000000002, 0.85])
```
"""
struct PolicySampler <: Sampler
    policy::Policy
    updater::Updater
    n::Integer
    rng::AbstractRNG
    verbose::Bool
end

function PolicySampler(
    pomdp::POMDP;
    policy::Policy=RandomPolicy(pomdp), 
    updater::Updater=DiscreteUpdater(pomdp), 
    n=10,
    rng::AbstractRNG=Random.GLOBAL_RNG,
    verbose::Bool=false
)
    PolicySampler(policy, updater, n, rng, verbose)
end


function (s::PolicySampler)(pomdp::POMDP)
    B = []
    mdp = GenerativeBeliefMDP(pomdp, s.updater)
    if s.verbose
        progress = Progress(s.n)
    end
    i = 0
    while true
        b = initialstate(mdp).val
        for _ in 1:s.n
            i += 1
            if i == s.n
                return unique!(B)
            end
            a = action(s.policy, b)
            # NOTE: backwards compatible
            # if isterminal(mdp.pomdp, rand(s.rng, b))
            #     break
            # end
            b = @gen(:sp)(mdp, b, a, s.rng)
            push!(B, b)
            if s.verbose
                next!(progress)
            end
        end
    end
    return B
end


function (s::PolicySampler)(pomdp::CircularMaze)
    B = []
    mdp = GenerativeBeliefMDP(pomdp, s.updater) 
    if s.verbose
        progress = Progress(s.n)
    end
    i = 0
    while true
        b = initialstate(mdp).val
        for _ in 1:s.n
            i += 1
            if i == s.n
                return unique!(B)
            end
            a = action(s.policy, b)
            b = @gen(:sp)(mdp, b, a, s.rng)
            if CMAZE_TERMINAL_FLAG
                global CMAZE_TERMINAL_FLAG = false
                break
            else
                push!(B, b)
                if s.verbose
                    next!(progress)
                end
            end
        end
    end
    return B
end






"""
    ExplorationPolicySampler

Samples belief states by rolling out an `ExplorationPolicy`. Essentially identical to `PolicySampler`.

## Fields
- `explorer::ExplorationPolicy`: The `ExplorationPolicy` used for decision making.
- `on_policy::Policy`: The fallback `Policy` used for decision making when not exploring.
- `updater::Updater`: The updater used for updating beliefs.
- `n::Integer`: The maximum number of simulated steps.
- `rng::AbstractRNG`: The random number generator used for sampling.
- `verbose::Bool`: Whether to use a progress bar while sampling.

## Constructors
    ExplorationPolicySampler(pomdp::POMDP; rng::AbstractRNG=Random.GLOBAL_RNG,
    explorer::ExplorationPolicy=EpsGreedyPolicy(pomdp, 0.1; rng=rng), on_policy=RandomPolicy(pomdp),
    updater::Updater=DiscreteUpdater(pomdp), n::Integer=10)

## Methods
    (s::ExplorationPolicySampler)(pomdp::POMDP)

Returns a vector of _unique_ belief states.

## Example Usage

```julia-repl
julia> pomdp = TigerPOMDP()
julia> sampler = ExplorationPolicySampler(pomdp; n=30)
julia> sampler(pomdp)
3-element Vector{Any}:
 DiscreteBelief{TigerPOMDP, Bool}(TigerPOMDP(-1.0, -100.0, 10.0, 0.85, 0.95), Bool[0, 1], [0.5, 0.5])
 DiscreteBelief{TigerPOMDP, Bool}(TigerPOMDP(-1.0, -100.0, 10.0, 0.85, 0.95), Bool[0, 1], [0.85, 0.15000000000000002])
 DiscreteBelief{TigerPOMDP, Bool}(TigerPOMDP(-1.0, -100.0, 10.0, 0.85, 0.95), Bool[0, 1], [0.9697986577181208, 0.030201342281879207])
```
"""
struct ExplorationPolicySampler <: Sampler
    explorer::ExplorationPolicy
    on_policy::Policy
    updater::Updater
    n::Integer
    rng::AbstractRNG
    verbose::Bool
end

function ExplorationPolicySampler(pomdp::POMDP; 
    rng::AbstractRNG=Random.GLOBAL_RNG,
    explorer::ExplorationPolicy=EpsGreedyPolicy(pomdp, 0.1; rng=rng),
    on_policy=RandomPolicy(pomdp),
    updater::Updater=DiscreteUpdater(pomdp), 
    n=10,
    verbose=false
)
    @assert n > 0 "n must be a positive integer"
    sampler = ExplorationPolicySampler(explorer, on_policy, updater, n, rng, verbose)
    return sampler
end



function (s::ExplorationPolicySampler)(pomdp::POMDP)
    B = []
    mdp = GenerativeBeliefMDP(pomdp, s.updater)
    if s.verbose
        progress = Progress(s.n)
    end
    i = 0
    while true
        b = initialstate(mdp).val
        for k in 1:s.n
            i += 1
            if i == s.n
                return unique!(B)
            end
            a = action(s.explorer, s.on_policy, k, b)
            # NOTE: backwards compatible
            # if isterminal(mdp.pomdp, rand(s.rng, b))
            #     break
            # end
            b = @gen(:sp)(mdp, b, a, s.rng)
            push!(B, b)
            if s.verbose
                next!(progress)
            end
        end
    end
    return B
end


function (s::ExplorationPolicySampler)(pomdp::CircularMaze)
    B = []
    mdp = GenerativeBeliefMDP(pomdp, s.updater)
    if s.verbose
        progress = Progress(s.n)
    end
    i = 0
    while true
        b = initialstate(mdp).val
        for k in 1:s.n
            i += 1
            if i == s.n
                return unique!(B)
            end
            a = action(s.explorer, s.on_policy, k, b)
            b = @gen(:sp)(mdp, b, a, s.rng)
            if CMAZE_TERMINAL_FLAG
                global CMAZE_TERMINAL_FLAG = false
                break
            else
                push!(B, b)
                if s.verbose
                    next!(progress)
                end
            end
        end
    end
    return B
end



