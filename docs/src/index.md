
```@setup 1
using Plots,Graphs,RandomForests,Random
pyplot()

g = Graphs.grid([4,4])
p = Iterators.product(0.0:0.1:0.3, 0.0:0.1:0.3);
xloc = zeros(nv(g))
yloc = zeros(nv(g))
global i = 0
for (x,y) in p
  global i += 1
  xloc[i] = x
  yloc[i] = y  
end


graphplotobj = RFGraphPlot(g,xloc,yloc,[],15,3,:viridis,false,"","")
plot(graphplotobj)
savefig("ex_graph.svg")

rt = random_spanning_tree(g)
treeplotobj = RFGraphPlot(rt.tree,xloc,yloc,[(i in rf.roots) for i = 1:nv(g)],15,3,:viridis,false,"","")
plot(treeplotobj)
savefig("ex_tree.svg")

rng = MersenneTwister(12)
rf = random_forest(g,1.0,rng)
forestplotobj = RFGraphPlot(SimpleDiGraph(rf),xloc,yloc,[(i in rf.roots) for i = 1:nv(g)],15,3,:viridis,false,"","")
plot(forestplotobj)
savefig("ex_forest.svg")

```

```@setup 2
using Graphs,RandomForests,PyPlot,Random
g = Graphs.grid([4,4])
p = Iterators.product(0.0:0.1:0.3, 0.0:0.1:0.3);
xloc = zeros(nv(g))
yloc = zeros(nv(g))
global i = 0
for (x,y) in p
  global i += 1
  xloc[i] = x
  yloc[i] = y  
end

rng = MersenneTwister(12)
rf = random_forest(g,0.1,rng)
forestplotobj = RFGraphPlot(SimpleDiGraph(rf),xloc,yloc,[(i in rf.roots) for i = 1:nv(g)],15,3,:viridis,false,"","")
plot(forestplotobj)
savefig("q=0.1.svg")

rf = random_forest(g,1.0,rng)
forestplotobj = RFGraphPlot(SimpleDiGraph(rf),xloc,yloc,[(i in rf.roots) for i = 1:nv(g)],15,3,:viridis,false,"","")
plot(forestplotobj)
savefig("q=1.0.svg")

rf = random_forest(g,5.0,rng)
forestplotobj = RFGraphPlot(SimpleDiGraph(rf),xloc,yloc,[(i in rf.roots) for i = 1:nv(g)],15,3,:viridis,false,"","")
plot(forestplotobj)
savefig("q=5.0.svg")


```

```@setup 3
using Graphs,RandomForests,Plots,Random
g = Graphs.grid([4,4])
p = Iterators.product(0.0:0.1:0.3, 0.0:0.1:0.3);
xloc = zeros(nv(g))
yloc = zeros(nv(g))
global i = 0
for (x,y) in p
  global i += 1
  xloc[i] = x
  yloc[i] = y  
end
rng = MersenneTwister(12)
rf = random_forest(g,1.0,rng)
forestplotobj = RFGraphPlot(SimpleDiGraph(rf),xloc,yloc,repeat([1.0],nv(g)),15,3,:jet,false,"","")
plot(forestplotobj)
savefig("quniform.svg")

idx = [1,4,13,16]
q = 0.02*ones(nv(g))
q[idx] .= (16-12*0.02)/4
rf = random_forest(g,q,rng)
rf = random_forest(g,1.0,rng)
forestplotobj = RFGraphPlot(SimpleDiGraph(rf),xloc,yloc,repeat([1.0],nv(g)),15,3,:jet,false,"","")
plot(forestplotobj)
savefig("qnonuniform.svg")

```
# RandomForests.jl: a Julia package for Random Forests on Graphs, and Applications

## Welcome to the Documentation of RandomForest.jl
A random spanning forest (RSF) is a special random object on graph/networks which has elegant theoretical links with graph Laplacians and wide set of applications in graph signal processing and machine learning.  

This package is dedicated to implementing:
- Sampling algorithms for RSFs
- Randomized algorithms for various problems including:
  - Graph Signal Interpolation and Tikhonov regularization
  - Trace Estimation
  - And more to come...

!!! warning
    The random forests produced by this package come from graph theory and are unrelated to the random forests found in machine learning.

## Installation Instructions
The package is not registered yet. Therefore, you can install it as follows:
```@julia
julia> ] add https://gricad-gitlab.univ-grenoble-alpes.fr/barthesi/RandomForests.jl
```

## Table of Contents
Apart from the usual documentation for the types and functions, [some theoretical materials](./rsf.md) along with [practical examples](./gtr.md) are provided to illustrate the full scope of `RandomForest.jl`.

```@contents
Pages = ["rsf.md","gtr.md","trace.md","typesandfunc.md"]
Depth = 5
```
