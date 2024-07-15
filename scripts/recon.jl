using Plots
using Random
using MultivariateStats

using POMDPs
using POMDPTools
using CompressedBeliefMDPs

Random.seed!(1)

function is_terminal_belief(row)
    return all(x -> x == 0, row[1:end-1]) && row[end] == 1
end

function plot_beliefs(original, reconstructed)
    x = 1:length(original)
    plot(x, original, label="Original Belief", linestyle=:solid, linewidth=2)
    plot!(x, reconstructed, label="Reconstructed Belief", linestyle=:dash, linewidth=2)
    xlabel!("State")
    ylabel!("Probability")
    title!("An Example Belief and Reconstruction")
end

function main()
    pomdp = CircularMaze(8, 25)
    sampler = PolicySampler(pomdp, n=500)
    B = sampler(pomdp)

    compressor = PCACompressor(100)

    B_numerical = mapreduce(b->convert_s(AbstractArray{Float64}, b, pomdp), hcat, B)' |> Matrix
    B_numerical = filter(row -> !is_terminal_belief(row), eachrow(B_numerical))  # exclude belief in terminal state
    B_numerical = reduce(hcat, B_numerical)'
    B_numerical = B_numerical[:, 1:end - 1]
    fit!(compressor, B_numerical)
    B̃ = compressor(B_numerical)
    ϕ = Dict(unique(t->t[2], zip(B, eachrow(B̃))))
    B_recon1 = reconstruct(compressor.M, B̃')'

    plots_dir = "plots"
    if !isdir(plots_dir)
        mkdir(plots_dir)
    end

    @show size(compressor.M)

    for i in 1:size(B_numerical, 1)
        original = B_numerical[i, :]
        reconstructed = B_recon1[i, :]
        plot_beliefs(original, reconstructed)
        savefig(joinpath(plots_dir, "belief_plot_$i.png"))
    end
end

main()
