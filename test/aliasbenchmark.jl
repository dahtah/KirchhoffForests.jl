using BenchmarkTools,SimpleWeightedGraphs,Graphs,KirchoffForests

g = grid([50,50])
gw = SimpleWeightedGraph(g)
gwp = PreprocessedWeightedGraph(gw)
#=
TODO: describe the test
=#


println("Simple Graph")
g = grid([50, 50])
# g = erdos_renyi(400,0.1)
display(@benchmark random_forest($g, .1))
println("")

println("Weighted Graph")
gw = SimpleWeightedGraph(g)
display(@benchmark random_forest($gw, .1))
println("")

println("Preprocessed Weighted Graph")
gwp = PreprocessedWeightedGraph(gw)
display(@benchmark random_forest($gwp, .1))
println("")
