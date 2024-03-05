using CompressedBeliefMDPs
using ExpFamilyPCA

using POMDPModels
using POMDPTools
using POMDPs

using NearestNeighbors, StaticArrays
using LocalFunctionApproximation
using LocalApproximationValueIteration

# pomdp = BabyPOMDP() 
pomdp = TigerPOMDP()

# make sampler
sampler = DiscreteRandomSampler(pomdp, n_samples=20)

# make compressor
n_components = 1
n_states = length(states(pomdp))
compressor = PoissonPCA(n_components, n_states)

# make updater
updater

solver = CompressedSolver(pomdp, compressor; verbose=true, max_iterations=1000)


# make solver
solver = LocalApproximationValueIterationSolver(func_approx, verbose=true, max_iterations=1000, is_mdp_generative=true, n_generative_samples=10)

# make compressed belief MDP
updater = DiscreteUpdater(pomdp)
mdp = CompressedBeliefMDP(pomdp, updater, compressor)

# solve
policy = solve(solver, mdp)