using CompressedBeliefMDPs
using ExpFamilyPCA

using POMDPs
using POMDPModels
using POMDPTools

using Distances

pomdp = TMaze(50, 0.99)
sampler = DiscreteRandomSampler(pomdp)
B = sample(sampler, pomdp; n_samples=100)
compressor = PoissonPCA(3, length(states(pomdp)))
B̃ = compress(compressor, B; verbose=true, maxiter=20)

# normalize
B_recon = decompress(compressor, B̃)
B_recon = abs.(B_recon)
B_recon ./= sum(B_recon, dims=2)
KL = mean([kl_divergence(p, q) for (p, q) in zip(eachrow(B), eachrow(B_recon))])
