@testset "aliascomparison" begin

    n = 50
    q = 1.0
    U = sprand(n,n,.5)
    g = SimpleWeightedGraph(U+U')
    gp = PreprocessedWeightedGraph(g)
    time_sum(g,q,trial) = @elapsed begin
        for i = 1 : trial
            random_forest(g,q)
        end
    end
    trial = 1000
    @test time_sum(g,q,trial) > time_sum(gp,q,trial)
end
