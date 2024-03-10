using Revise

using CompressedBeliefMDPs

using POMDPs
using POMDPModels
using POMDPTools

# using ExpFamilyPCA

# TODO: test w/ Kalman filters 

pomdp = TMaze(47, 0.99)
# pomdp = TigerPOMDP()
# pomdp = BabyPOMDP()

# sampler = DiscreteEpsGreedySampler(pomdp, k->0.05*0.9^(k/10))  # TODO: debug
# sampler = DiscreteEpsGreedySampler(pomdp, 0.05)  # TODO: debug
sampler = DiscreteRandomSampler(pomdp)
compressor = PCA(2)  # TODO: debug
# compressor = PoissonPCA(2, length(states(pomdp)))
solver = CompressedSolver(pomdp, sampler, compressor; n_samples=5)
approx_policy = solve(solver, pomdp; verbose=true, max_iterations=10)

# TODO: benchmark against QMDP
s = initialstate(pomdp)
v = value(approx_policy, s)  # returns the approximately optimal value for state s
a = action(approx_policy, s) # returns the approximately optimal action for state s