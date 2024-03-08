@testset "aliascomparison" begin
    #=
    TODO:
    - describe the test
    - add "using/import"s
    =#

    n = 50
    q = 1.0
    U = sprand(n ,n, .5)
    g = SimpleWeightedGraph(U + U')
    gp = PreprocessedWeightedGraph(g)

    time_sum(g, q, trial) = @elapsed begin
        for i in 1:trial
            random_forest(g, q)
        end
    end

    trial = 1000
    @assert time_sum(g, q, trial) > time_sum(gp, q, trial)
end
