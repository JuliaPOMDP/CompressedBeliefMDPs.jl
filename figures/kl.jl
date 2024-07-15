using Revise
using Infiltrator

using Plots
using Random
using MultivariateStats

using POMDPs
using POMDPTools
using CompressedBeliefMDPs

Random.seed!(1)

pomdp = CircularMaze(5, 100)
# pomdp = TMaze()
sampler = PolicySampler(pomdp, n=500)
compressor = PCACompressor(10)

# get beliefs
B = sampler(pomdp)

# get compressed beliefs
B_numerical = mapreduce(b->convert_s(AbstractArray{Float64}, b, pomdp), hcat, B)' |> Matrix
B_numerical = B_numerical[:,1:end-1]  # ignore belief in TerminalState 
fit!(compressor, B_numerical)
B̃ = compressor(B_numerical)
B_recon = reconstruct(compressor.M, B̃')'

# TODO: add comparison to our reconstruction

function plot_beliefs(original, reconstructed)
    # Define x-axis (assuming the states are ordered sequentially)
    x = 1:length(original)
    
    # Plot the beliefs
    plot(x, original, label="Original Belief", linestyle=:solid, linewidth=2)
    plot!(x, reconstructed, label="Reconstructed Belief", linestyle=:dash, linewidth=2)
    
    # Add labels and title
    xlabel!("State")
    ylabel!("Probability")
    title!("An Example Belief and Reconstruction")
end

# Loop through indices and save each plot in a folder
plots_dir = "plots"
if !isdir(plots_dir)
    mkdir(plots_dir)
end

@show size(compressor.M)

for i in 1:size(B_numerical, 1)
    original = B_numerical[i, :]
    reconstructed = B_recon[i, :]
    plot_beliefs(original, reconstructed)
    savefig(joinpath(plots_dir, "belief_plot_$i.png"))
end
