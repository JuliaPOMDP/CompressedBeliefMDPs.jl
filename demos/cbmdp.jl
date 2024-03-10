using Revise
using CompressedBeliefMDPs

using POMDPs
using POMDPModels
using POMDPTools


pomdp = TMaze(50, 0.99)
CompressedBeliefMDP(pomdp, compressor, sampler)
