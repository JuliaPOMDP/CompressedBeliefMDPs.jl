function test_compressor(C::Function, maxoutdim::Int)
    pomdp = BabyPOMDP()  # TODO: change to TMaze once I figure out how to properly sample
    compressor = C(maxoutdim)
    solver = CompressedBeliefSolver(pomdp; compressor=compressor, n=20)
    policy = solve(solver, pomdp)
    s = initialstate(pomdp)
    _ = action(policy, s)
    _ = value(policy, s)
    return policy
end

MV_STATS_COMPRESSORS = (
    PCACompressor,
    KernelPCACompressor,
    PPCACompressor,
    FactorAnalysisCompressor,
    MDSCompressor
)

@testset "Compressor Tests" begin
    @testset "$C" for C in MV_STATS_COMPRESSORS
        @test_nowarn test_compressor(C, 1)
        @test_nowarn test_compressor(C, 2)
    end
end
