using Revise
using Infiltrator

using Plots
using Random
using MultivariateStats
using Statistics
using Distances

using POMDPs
using POMDPTools
using CompressedBeliefMDPs

Random.seed!(1)

function is_terminal_belief(row)
    return all(x -> x == 0, row[1:end-1]) && row[end] == 1
end

function normalize_rows(M)
    normalized = clamp.(M, 0, 1)
    row_sums = sum(normalized, dims=2)
    normalized = normalized ./ row_sums
    return normalized
end

function calculate_l2_distance(X, Y)
    X = normalize_rows(X)
    Y = normalize_rows(Y)
    l2 = colwise(Euclidean(), X', Y')
    return l2
end

function plot_l2_distance(mean_l2, std_l2, max_bases)
    errorbar = plot(
        max_bases, mean_l2, 
        yerr=std_l2, 
        label="Mean L2 Distance", 
        marker=:x, 
        line=:dash
    )
    xlabel!("Maximum Number of Bases")
    ylabel!("Average L2 Distance")
    title!("Average L2 Distance vs. Maximum Number of Bases")
    display(errorbar)
    savefig(errorbar, "average_l2_distance_vs_max_bases.png")
end

function main()
    pomdp = CircularMaze(8, 25)
    sampler = PolicySampler(pomdp, n=500)
    B = sampler(pomdp)

    means = []
    stds = []

    for maxoutdim in 1:30
        compressor = PCACompressor(maxoutdim)
        B_numerical = mapreduce(b->convert_s(AbstractArray{Float64}, b, pomdp), hcat, B)' |> Matrix
        B_numerical = filter(row -> !is_terminal_belief(row), eachrow(B_numerical))  # exclude belief in terminal state
        B_numerical = reduce(hcat, B_numerical)'
        B_numerical = B_numerical[:, 1:end - 1]
        fit!(compressor, B_numerical)
        B̃ = compressor(B_numerical)
        B_recon = reconstruct(compressor.M, B̃')'
        l2 = calculate_l2_distance(B_numerical, B_recon)
        mean_l2 = mean(l2)
        std_l2 = std(l2)
        push!(means, mean_l2)
        push!(stds, std_l2)
    end

    plot_l2_distance(means, stds, 1:30)
end

main()
