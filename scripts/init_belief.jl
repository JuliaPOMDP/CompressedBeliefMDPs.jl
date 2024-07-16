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

function plot_beliefs(original)
    x = 1:length(original)
    plot(x, original, linestyle=:solid, linewidth=2, legend=false)
    xlabel!("State")
    ylabel!("Probability")
    title!("A Sample Belief")
end

function main()
    pomdp = CircularMaze(2, 100)
    sampler = PolicySampler(pomdp, n=10)
    B = sampler(pomdp)

    B_numerical = mapreduce(b->convert_s(AbstractArray{Float64}, b, pomdp), hcat, B)' |> Matrix
    B_numerical = filter(row -> !is_terminal_belief(row), eachrow(B_numerical))  # exclude belief in terminal state
    B_numerical = reduce(hcat, B_numerical)'
    B_numerical = B_numerical[:, 1:end - 1]

    original = B_numerical[1, :]
    plot_beliefs(original)
    savefig("initial_belief.png")
end

main()
