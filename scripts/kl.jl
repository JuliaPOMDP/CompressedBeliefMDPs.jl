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

function calculate_kl_divergence(X, Y)
    X = normalize_rows(X)
    Y = normalize_rows(Y)
    kl = colwise(KLDivergence(), X', Y')
    return kl
end

function plot_kl_divergence(mean_kl, std_kl, max_bases)
    errorbar = plot(
        max_bases, mean_kl, 
        yerr=std_kl, 
        label="Mean KL Divergence", 
        marker=:x, 
        line=:dash
    )
    xlabel!("Maximum Number of Bases")
    ylabel!("Average KL Divergence")
    title!("Average KL Divergence vs. Maximum Number of Bases")
    display(errorbar)
    savefig(errorbar, "average_kl_divergence_vs_max_bases.png")
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
        kl = calculate_kl_divergence(B_numerical, B_recon)
        mean_kl = mean(kl)
        std_kl = std(kl)
        push!(means, mean_kl)
        push!(stds, std_kl)
    end

    plot_kl_divergence(means, stds, 1:30)
end

main()
