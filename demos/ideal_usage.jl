using CompressedBeliefMDPs

using POMDPModels
using POMDPs

using ExpFamilyPCA



pomdp = TigerPOMDP()
# sampler = DiscreteEpsGreedySampler(pomdp, k->0.05*0.9^(k/10))  # TODO: debug for other samplers
sampler = DiscreteRandomSampler(pomdp)
# compressor = PCA(2)
compressor = PoissonPCA(1, length(states(pomdp)))
# compressor = PoissonPCA(2, length(states(pomdp)))  # TODO: debug dimension errors
solver = CompressedSolver(pomdp, sampler, compressor; n_samples=5)
approx_policy = solve(solver, pomdp; verbose=true, max_iterations=10)

s = initialstate(pomdp)
# v = value(approx_policy, s)  # returns the approximately optimal value for state s
a = action(approx_policy, s) # returns the approximately optimal action for state s