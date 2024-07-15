@testset "Solver/Sampler Tests" begin
    @testset "Discrete S, A, O" begin
        discrete_pomdps = [
            BabyPOMDP(),
            TigerPOMDP(),
        ]
        @testset "$pomdp" for pomdp in discrete_pomdps
            @testset "PolicySampler" begin
                sampler = PolicySampler(pomdp)
                @test_nowarn test_solver(CompressedBeliefSolver(pomdp; sampler=sampler), pomdp)
            end
            @testset "ExplorationPolicySampler" begin
                sampler = ExplorationPolicySampler(pomdp)
                @test_nowarn test_solver(CompressedBeliefSolver(pomdp; sampler=sampler), pomdp)
            end
            @testset "BeliefExpansionSampler" begin
                sampler = BeliefExpansionSampler(pomdp)
                @test_nowarn test_solver(CompressedBeliefSolver(pomdp; sampler=sampler), pomdp)
            end
        end
    end
    
    @testset "Continuous S, A" begin
        continuous_pomdps = [
            LightDark1D(),
        ]
        @testset "$pomdp" for pomdp in continuous_pomdps
            # NOTE: have to use a different base solver since LocalApproximationValueIterationSolver only supports discrete S, A
            # TODO: check if MCTS is appropriate for belief compression
            base_solver = MCTSSolver(n_iterations=10, depth=5, exploration_constant=5.0)
            updater = BootstrapFilter(pomdp, 100)
            solver = CompressedBeliefSolver(pomdp, base_solver; sampler=PolicySampler(pomdp; updater=updater), updater=updater)
            @test_nowarn test_solver(solver, pomdp)
        end 
    end
end