@testset "Solver Tests" begin
    compressor = PCACompressor(1)
    @testset "$pomdp" for pomdp in (BabyPOMDP(), TigerPOMDP(), TMaze(6, 0.99), LightDark1D())
        solver = CompressedBeliefSolver(pomdp; n=10)
        @test_nowarn test_solver(solver, pomdp)
    end
end

# TODO: add test w/ MCTS