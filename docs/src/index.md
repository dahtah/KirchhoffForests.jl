
```@setup 1
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
param = PlotParam(xloc,yloc,false,[],500,3,:viridis,false,"","")
figure()
plot_graph(g,param=param)
savefig("ex_graph.svg")

param.showRoots= true
param.cmap=:viridis
rt = random_spanning_tree(g)
figure()
plot_tree(rt,param=param)
savefig("ex_tree.svg")

param.showRoots= true
param.cmap=:viridis
rng = MersenneTwister(12)
rf = random_forest(g,1.0,rng)
figure()
plot_forest(rf,param=param)
savefig("ex_forest.svg")

```

```@setup 2
using Graphs,RandomForests,PyPlot
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
param = PlotParam(xloc,yloc,true,[],500,3,:viridis,false,"","")
param.showRoots= true
param.cmap=:viridis
rf = random_forest(g,0.1)
figure()
plot_forest(rf,param=param)
savefig("q=0,1.svg")

rf = random_forest(g,1)
figure()
plot_forest(rf,param=param)
savefig("q=1.svg")

rf = random_forest(g,5.0)
figure()
plot_forest(rf,param=param)
savefig("q=5.svg")

```

```@setup 3
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
param = PlotParam(xloc,yloc,false,[],500,3,:viridis,true,"q","")
param.signal = ones(nv(g))
param.cmap=:jet
rf = random_forest(g,1.0,rng)
figure()
plot_forest(rf,param=param)
savefig("quniform.svg")

idx = [1,4,13,16]
param.signal = 0.02*ones(nv(g))
param.signal[idx] .= (16-12*0.02)/4
param.cmap=:jet
rf = random_forest(g,param.signal,rng)
figure()
plot_forest(rf,param=param)
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
