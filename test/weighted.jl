@testset "weighted" begin
    import Random.seed!
    using SparseArrays
    seed!(1)
    gg = SimpleWeightedGraph(grid([4,4]))
    for i in 1:100
        check_correctness(random_forest(gg,i/10))
    end
    U = sprand(5,5,.5)
    gg = SimpleWeightedGraph(U+U')
    for i in 1:100
        check_correctness(random_forest(gg,i/10))
    end
    U = sprand(5,5,.5)
    gg = PreprocessedWeightedGraph(U+U')
    for i in 1:100
        check_correctness(random_forest(gg,i/10))
    end
end
