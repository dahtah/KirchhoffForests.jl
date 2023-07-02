module RandomForests
using Graphs,LinearAlgebra,SparseArrays,SimpleWeightedGraphs,RecipesBase
import StatsBase.denserank,Statistics.mean,Base.show,Base.sum,
StatsBase.counts, StatsBase.quantile, Distributions.Normal
import Random.MersenneTwister,Random.rand!
import Graphs.SimpleDiGraph,Graphs.nv,Graphs.ne,Graphs.degree,Graphs.outneighbors
import SimpleWeightedGraphs.AbstractSimpleWeightedGraph
import Graphs:
nv,ne,outneighbors,is_directed,inneighbors


export random_forest,smooth,smooth_rf,smooth_rf_adapt,RandomForest,
      SimpleDiGraph,nroots,next,Partition,PreprocessedWeightedGraph
export newton_poisson_noise,irls,admm_edge_lasso,SURE
export smooth_rf_xbar,smooth_rf_xtilde

export reduced_graph,smooth_ms
export self_roots
export random_successor
export random_spanning_tree
export plot_graph,plot_tree,plot_forest,PlotParam,grid_layout

export root_boundary_track, partition_boundary_track, trace_estimator

include("alias.jl")


"""
    rf = RandomForest(next::Array{Int},
                      roots :: BitSet,
                      nroots :: Int,
                      root :: Array{Int,1})

A simple struct for a rooted spanning forest for given arguments.

# Arguments
* ```next```: a vector of vertices where ```next[i] = j``` means that node ``i`` points to node ``j`` in the forest. If ```next[i] = 0```, ``i`` is a root.
* ```roots```: the set of roots in the forest.
* ```nroots```: the number of the roots.
* ```root```: a vector of vertices where ```root[i] = j``` means that node ``i`` is rooted in node ``j`` in the forest.
"""
struct RandomForest
    next :: Array{Int}
    roots :: BitSet
    nroots :: Int
    root :: Array{Int,1}
end

function show(io::IO, rf::RandomForest)
    println(io, "Random forest. Size of original graph $(nv(rf)).")
    println(io,"Number of trees $(nroots(rf))")
end

function nv(rf::RandomForest)
    length(rf.next)
end

function nroots(rf::RandomForest)
    rf.nroots
end

function ne(rf::RandomForest)
    nv(rf)-nroots(rf)
end

"""
    v = next(rf::RandomForest)

Return a vector of indices v, where ```v[i] = j``` means that node ``i`` points to node ``j`` in the forest. If ```v[i] = 0```, i is a root.
"""
function next(rf::RandomForest)
    rf.next
end

function outneighbors(rf::RandomForest,i)
    rf.next[i] > 0 ? [rf.next[i]] : Array{Int64,1}()
end

"""
    random_forest(G::AbstractGraph,q[,B::AbstractVector])

Run Wilson's algorithm on G to generate a random forest with parameter ``q``.

# Arguments
* ```g```: Input graph
* ```q```: Parameter that determines the probability that the random walk is interrupted at a node. If q is a scalar, that probability equals ```q/(q+d[i])``` at node i with degree ```d[i]```. If q is a vector, it equals ``` q[i]/(q[i]+d[i]) ```.
* ```B```: the set of predefined rootset. Random walks are interrupted whenever they reach a node in ```B```.

# Example

```jldoctest
using Graphs,RandomForests

julia>  G = grid([3,3])
{9, 12} undirected simple Int64 graph

julia> random_forest(G,.4) # Scalar q
Random forest. Size of original graph 9.
Number of trees 4

julia> random_forest(G,rand(nv(G))) # q varies over vertices
Random forest. Size of original graph 9.
Number of trees 2

julia> B = rand(1:nv(G),3) # A predefined rootset.
3-element Array{Int64,1}:
 3
 7
 8

julia> rf = random_forest(G,rand(nv(G)),B)
Random forest. Size of original graph 9.
Number of trees 4

julia> next(rf)
9-element Array{Int64,1}:
 4
 1
 0
 7
 2
 3
 0
 0
 0

julia> rf.roots
BitSet with 4 elements:
 3
 7
 8
 9
```
"""
function random_forest(G::AbstractGraph,q::Number)
    roots = BitSet()
    root = zeros(Int64,nv(G))
    nroots = Int(0)
    d = degree(G)
    n = nv(G)
    in_tree = falses(n)
    next = zeros(Int64,n)
    @inbounds for i in 1:n
        u = Int64(i)
        while !in_tree[u]
            if (((q+d[u]))*rand() < q)
                in_tree[u] = true
                push!(roots,u)
                nroots+=1
                root[u] = u
                next[u] = 0
            else
                next[u] = random_successor(G,u)
                u = next[u]
            end
        end
        r = root[u]
        #Retrace steps, erasing loops
        u = i
        while !in_tree[u]
            root[u] = r
            in_tree[u] = true
            u = next[u]
        end
    end
    RandomForest(next,roots,nroots,root)
end

function random_forest(G::AbstractGraph,q::Number,B::AbstractVector)
    roots = BitSet()
    root = zeros(Int64,nv(G))
    nroots = Int(0)
    d = degree(G)
    n = nv(G)
    in_tree = falses(n)
    next = zeros(Int64,n)
    @inbounds for i in 1:n
        u = Int64(i)
        while !in_tree[u]
            if ((((q+d[u]))*rand() < q) || u in B)
                in_tree[u] = true
                push!(roots,u)
                nroots+=1
                root[u] = u
                next[u] = 0
            else
                next[u] = random_successor(G,u)
                u = next[u]
            end
        end
        r = root[u]
        #Retrace steps, erasing loops
        u = i
        while !in_tree[u]
            root[u] = r
            in_tree[u] = true
            u = next[u]
        end
    end
    RandomForest(next,roots,nroots,root)
end


function random_forest(G::AbstractGraph,q::AbstractVector)
    @assert length(q)==nv(G)
    roots = BitSet()
    root = zeros(Int64,nv(G))
    nroots = Int(0)

    n = nv(G)

    in_tree = falses(n)
    next = zeros(Int64,n)
    @inbounds for i in 1:n
        u = Int64(i)

        while !in_tree[u]
            if (rand() < q[u]/(q[u]+degree(G,u)))
                in_tree[u] = true
                push!(roots,u)
                nroots+=1
                root[u] = u
                next[u] = 0
            else
                next[u] = random_successor(G,u)
                u = next[u]
            end
        end
        r = root[u]

        #Retrace steps, erasing loops
        u = i
        while !in_tree[u]
            root[u] = r
            in_tree[u] = true
            u = next[u]
        end
    end
    RandomForest(next,roots,nroots,root)
end

function random_forest(G::AbstractGraph,q::AbstractVector,B::AbstractVector)
    @assert length(q)==nv(G)
    roots = BitSet()
    root = zeros(Int64,nv(G))
    nroots = Int(0)

    n = nv(G)

    in_tree = falses(n)
    next = zeros(Int64,n)
    @inbounds for i in 1:n
        u = Int64(i)

        while !in_tree[u]
            if ((rand() < q[u]/(q[u]+degree(G,u))) || (u in B))
                in_tree[u] = true
                push!(roots,u)
                nroots+=1
                root[u] = u
                next[u] = 0
            else
                next[u] = random_successor(G,u)
                u = next[u]
            end
        end
        r = root[u]

        #Retrace steps, erasing loops
        u = i
        while !in_tree[u]
            root[u] = r
            in_tree[u] = true
            u = next[u]
        end
    end
    RandomForest(next,roots,nroots,root)
end


"""
    random_successor(G::AbstractGraph,i :: T) where T <: Int64

Return a neighbor j of node i in the graph g with probability `` \\frac{w(i,j)}{d_i+q} ``. If the graph is preprocessed (see [`PreprocessedWeightedGraph`](@ref)), the cost is constant.
"""
function random_successor(G::AbstractGraph,i :: T) where T <: Int64
    nbrs = neighbors(G, i)
    rand(nbrs)
end

function random_successor(g :: SimpleWeightedGraph,i :: T) where T <: Int64
    W = weights(g)
    rn = W.colptr[i]:(W.colptr[i+1]-1)
    w = W.nzval[rn]
    w /= sum(w)
    u = rand()
    j = 0
    s = 0
    while s < u && j < length(w)
        s+= w[j+1]
        j+=1
    end
    W.rowval[rn[1]+j-1]
end

function random_successor(g :: PreprocessedWeightedGraph,i :: T) where T <: Int64

    sample = alias_draw(g,i)
    sample
end


"""
    SimpleDiGraph(rf :: RandomForest)

Convert a RandomForest rf to a SimpleDiGraph.

# Example

```jldoctest
julia> using Graphs, RandomForests

julia> g = grid([3,3])
{9, 12} undirected simple Int64 graph

julia> rf = random_forest(g,.4)
Random forest. Size of original graph 9.
Number of trees 4


julia> f = SimpleDiGraph(rf)
{9, 5} directed simple Int64 graph

julia> connected_components(f)
4-element Array{Array{Int64,1},1}:
 [1, 4]
 [2, 3, 5, 6]
 [7, 8]
 [9]
```

"""
function SimpleDiGraph(rf :: RandomForest)
    n = length(rf.next)
    ff = SimpleDiGraph(n)
    for i in 1:n
        (rf.next[i] > 0) && add_edge!(ff,i,rf.next[i])
    end
    ff
end

include("random_forest_with_seeds.jl")
include("random_spanning_tree.jl")
include("smoothing.jl")
include("moments.jl")
include("multiscale.jl")
include("optimization.jl")
include("paramselection.jl")
include("efficient_smoothing.jl")
include("plotting_recipe.jl")
include("trace.jl")
end # module
