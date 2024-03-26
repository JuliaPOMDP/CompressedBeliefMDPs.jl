@testset "Solver Tests" begin
    compressor = PCACompressor(1)
    @testset "Discrete S, A" begin
        @testset "$pomdp" for pomdp in (BabyPOMDP(), TigerPOMDP(), TMaze(6, 0.99))
            solver = CompressedBeliefSolver(pomdp; n=10)
            @test_nowarn test_solver(solver, pomdp)
        end
    end
    @testset "Continuous S, A" begin
        @testset "$pomdp" for pomdp in (LightDark1D(),)
            base_solver = MCTSSolver(n_iterations=10, depth=5, exploration_constant=5.0)
            solver = CompressedBeliefSolver(pomdp, base_solver; n=10)
            @test_nowarn test_solver(solver, pomdp)
        end 
    end
end