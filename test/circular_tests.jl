let
    pomdp = CircularMaze(2, 5, 0.99)
    @test has_consistent_distributions(pomdp)

    # test non-exported solvers
    # solvers = [
    #     MCTSSolver(n_iterations=10, depth=5, exploration_constant=5.0),
    # ]
    # @testset "$solver" for solver in solvers
    #     @test test_solver(solver, pomdp)
    #     # @test_nowarn test_solver(solver, pomdp)
    # end

    # test CompressedBeliefSolver
    @testset "Samplers" begin
        @testset "PolicySampler" begin
            @test_nowarn test_solver(CompressedBeliefSolver(pomdp; sampler=PolicySampler(pomdp)), pomdp)
        end
        @testset "ExplorationPolicySampler" begin
            @test_nowarn test_solver(CompressedBeliefSolver(pomdp; sampler=PolicySampler(pomdp)), pomdp)
        end
        @testset "BeliefExpansionSampler" begin
            @test_nowarn test_solver(CompressedBeliefSolver(pomdp; sampler=PolicySampler(pomdp)), pomdp)
        end
    end
   

    # TODO: test simulations, RNG, rewards for non goal states

    @testset "Transition/Reward Tests" begin
        for s in pomdp.goals
            @test reward(pomdp, s, CMAZE_LEFT) == 0
            @test reward(pomdp, s, CMAZE_RIGHT) == 0
            @test reward(pomdp, s, CMAZE_SENSE_CORRIDOR) == 0
            @test reward(pomdp, s, CMAZE_DECLARE_GOAL) == pomdp.r_findgoal
        end

        s = pomdp.goals[1]
        sp = CircularMazeState(s.corridor, (s.x + 1) % pomdp.corridor_length)
        @test reward(pomdp, sp, CMAZE_LEFT) == 0
        @test reward(pomdp, sp, CMAZE_RIGHT) == 0
        @test reward(pomdp, sp, CMAZE_SENSE_CORRIDOR) == 0
        @test reward(pomdp, sp, CMAZE_DECLARE_GOAL) == 0
    end
end