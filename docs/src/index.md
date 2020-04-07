# RandomForests.jl: a Julia package for Random Forests on Graphs, and Applications

## What's a random forest? 

The random forests produced by this package come from graph theory, and are
unrelated to the random forests found in machine learning. A tree is a graph
without cycles, and a forest is a set of trees. 
We are interested in a specific way of generating random *spanning* forests in a
graph, because of its deep ties to the graph Laplacian. 

```@setup 1
using LightGraphs,TikzPictures,TikzGraphs
g = grid([2,2])
add_vertex!(g)
add_edge!(g,4,5)
t = TikzGraphs.plot(g)
save(SVG("ex_graph.svg"),t)

g2 = SimpleDiGraph(5)
add_edge!(g2,2,1)
add_edge!(g2,1,4)
add_edge!(g2,4,5)
add_edge!(g2,3,5)
t = TikzGraphs.plot(g2)
save(SVG("ex_tree.svg"),t)

g3 = SimpleDiGraph(5)
add_edge!(g3,2,3)
add_edge!(g3,1,4)
add_edge!(g3,4,5)

t = TikzGraphs.plot(g3)
save(SVG("ex_forest.svg"),t)
```

This is an example of a graph (with a loop):

![](ex_graph.svg)

This is an example of a spanning tree for the (same) graph:

![](ex_tree.svg)

Finally, this is an example of a spanning forest:

![](ex_forest.svg)

## Rooted spanning forests

Importantly, all the forests we use are considered to be *rooted*: each tree in
the forest is directed, and all edges point towards the root of the tree. In the
forest above, the roots are the nodes 5 and 3. 

When we talk about a "random spanning forest", we mean a forest $\phi$ sampled
from the following distribution: 
```math
p(\phi) = \frac{1}{z} q^{R(\phi)} \prod_{(ij)\in \phi} w_{ij}
```


where:

- $ \phi $ is a forest, viewed as a set of edges,
- $ R(\phi) $ is the number of trees in $\phi$,
- $ w_{ij} $ is the weight associated with edge (ij) (which equals 1 if the
  graph is unweighted)
- $ q $ is a parameter that determines the average number of trees. 
- $ z $ is an integration constant. 


## References

## Functions and types

```@autodocs
Modules = [RandomForests]
Order   = [:function, :type]
```
