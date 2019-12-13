using RandomForests
using LightGraphs,SimpleGraphs,SimpleWeightedGraphs
using SparseArrays
n = 50
q = 1.0
U = sprand(n,n,.5)
g = SimpleWeightedGraph(U+U')
gp = PreprocessedWeightedGraph(g)
time_sum(g,q,trial) = @time begin
    for i = 1 : trial
        random_forest(g,q)
    end
end
trial = 1000
println("RF sampling with naive algorithm")
time_sum(g,q,trial)
println("RF sampling with alias sampling")
time_sum(gp,q,trial)


j = Int64(rand(1:n))
trial = 100000

time_sum_suc(g,trial) = @time begin
    for i = 1 : trial
        random_successor(g,j)
    end
end
println("random_successor with naive algorithm with ",trial," trials")
time_sum_suc(g,trial)
println("random_successor with alias sampling with ",trial," trials")
time_sum_suc(gp,trial)
