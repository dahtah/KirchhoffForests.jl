using BenchmarkTools,SimpleWeightedGraphs,Graphs,RandomForests

g = grid([50,50])
# g = erdos_renyi(400,0.1)
gw = SimpleWeightedGraph(g)
gwp = PreprocessedWeightedGraph(gw)

println("Simple Graph")
display(@benchmark random_forest($g,.1))
println("")
println("Weighted Graph")
display(@benchmark random_forest($gw,.1))
println("")
println("Preprocessed Weighted Graph")
display(@benchmark random_forest($gwp,.1))
println("")
