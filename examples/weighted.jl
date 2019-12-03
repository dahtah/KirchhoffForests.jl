using SimpleWeightedGraphs, LightGraphs,RandomForests
g =SimpleWeightedGraph(grid([30,30]))
prepg = PreprocessedWeightedGraph(g)
rf= random_forest(prepg,.4)