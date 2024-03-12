function test_compressor(C::Function, maxoutdim::Int)
    pomdp = TMaze(20, 0.99)
    sampler = DiscreteRandomSampler(pomdp)
    compressor = C(maxoutdim)
    solver = CompressedSolver(pomdp, sampler, compressor; n_samples=20)
    approx_policy = solve(solver, pomdp; verbose=false, max_iterations=5)
    s = initialstate(pomdp)
    _ = value(approx_policy, s)
    _ = action(approx_policy, s)
    return approx_policy
end

MV_STATS_COMPRESSORS = (
    PCACompressor,
    KernelPCACompressor,
    PPCACompressor,
    # FactorAnalysisCompressor
)

@testset "Compressor Tests" begin
    @testset "$C" for C in MV_STATS_COMPRESSORS
        @inferred test_compressor(C, 1)
        @inferred test_compressor(C, 10)
    end
end
