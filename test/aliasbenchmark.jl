using BenchmarkTools,SimpleWeightedGraphs,LightGraphs,RandomForests

# g = grid([10,10])
g = erdos_renyi(400,0.1)
gw = SimpleWeightedGraph(g)
gwp = PreprocessedWeightedGraph(gw)

println(@benchmark random_forest($g,.1))
println(@benchmark random_forest($gw,.1))
println(@benchmark random_forest($gwp,.1))
