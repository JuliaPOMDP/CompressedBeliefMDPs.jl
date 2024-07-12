using Plots
using POMDPs
using POMDPTools
using CompressedBeliefMDPs


pomdp = CircularMaze(2, 50)
sampler = PolicySampler(pomdp)
B = sampler(pomdp)

belief = B[1]


