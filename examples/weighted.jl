using RandomForests
using SimpleWeightedGraphs, LightGraphs
g =SimpleWeightedGraph(grid([30,30]))
prepg = PreprocessedWeightedGraph(g)
rf= random_forest(prepg,1.0)
println(rf)
