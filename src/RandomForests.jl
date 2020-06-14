module RandomForests

using LinearAlgebra, SparseArrays
import Base:
    show,
    sum
import Statistics: mean
import StatsBase:
    counts,
    denserank

using LightGraphs
import LightGraphs:
    degree,
    inneighbors,
    is_directed,
    nv,
    ne,
    outneighbors,
    SimpleDiGraph

using SimpleWeightedGraphs
import SimpleWeightedGraphs:
    AbstractSimpleWeightedGraph

include("./alias.jl")

export
    random_forest,
    smooth,
    smooth_rf,
    smooth_rf_adapt,
    RandomForest,
    SimpleDiGraph,
    nroots,
    next,
    Partition,
    PreprocessedWeightedGraph,
    reduced_graph,
    smooth_ms,
    self_roots,
    random_successor,
    random_spanning_tree


#=
TODO:
- need clean up using and import
- Too many functions/methods are exposed, with potential overlaps
Suggestion: use a Python inspired approach "import nnumpy as np"
import LightGraphs; const LG = LightGraphs
to then call LG.whatever_function
=#

# abstract type AbstractSimpleWeightedGraph{T<:Integer,U<:Real}<:AbstractGraph{T} end
struct PreprocessedWeightedGraph{T<:Integer,U<:Real}<:AbstractSimpleWeightedGraph{T,U}
    weights::SparseMatrixCSC{U,T}
    K::SparseMatrixCSC{Int64,T}
    P::SparseMatrixCSC{Float64,T}
end

function PreprocessedWeightedGraph(adjmx::SparseMatrixCSC{U,T}) where {T<:Integer,U<:Real}
    K, P = alias_preprocess(SimpleWeightedGraph(adjmx))
    PreprocessedWeightedGraph{T, U}(adjmx,K,P)
end

## todo : not sure
PreprocessedWeightedGraph(g::LightGraphs.SimpleGraphs.SimpleGraph{T}, ::Type{U}=Float64) where {T<:Integer,U<:Real} =
    PreprocessedWeightedGraph(adjacency_matrix(g, U))
PreprocessedWeightedGraph(g::SimpleWeightedGraph) = PreprocessedWeightedGraph(g.weights)
inneighbors(g::PreprocessedWeightedGraph, v::Integer) = g.weights[v,:].nzind
is_directed(::Type{PreprocessedWeightedGraph}) = false
is_directed(::Type{PreprocessedWeightedGraph{T, U}}) where {T, U} = false
is_directed(g::PreprocessedWeightedGraph) = false
ne(g::PreprocessedWeightedGraph) = nnz(g.weights) / 2
degree(g::PreprocessedWeightedGraph) = sum(g.weights, dims=1)
degree(g::SimpleWeightedGraph) = sum(g.weights, dims=1)

struct RandomForest
    next::Array{Int}  # todo what size?
    roots::Set{Int}
    nroots::Int
    root::Array{Int,1}
end

function show(io::IO, rf::RandomForest)
    println(io, "Random forest. Size of original graph $(nv(rf)).")
    println(io, "Number of trees $(nroots(rf))")
end

nv(rf::RandomForest) = length(rf.next)
nroots(rf::RandomForest) = rf.nroots
ne(rf::RandomForest) = nv(rf) - nroots(rf)

"""
    next(rf::RandomForest)

Return a vector of indices v, where v[i] = j means that node i points to node j
in the forest. If v[i] = 0 i is a root.
"""
next(rf::RandomForest) = rf.next

outneighbors(rf::RandomForest,i) = rf.next[i] > 0 ? [rf.next[i]] : Array{Int64,1}()

"""
    random_forest(G::AbstractGraph,q)

Run Wilson's algorithm on G to generate a random forest with parameter "q". q determines the probability that
the random walk is interrupted at a node. If q is a scalar, that probability equals q/(q+d[i]) at node i with
degree d[i]. If q is a vector, it equals q[i]/(q[i]+d[i]).

# Example

```
using LightGraphs
G = grid([3,3])
random_forest(G,.4)
q_varying = rand(nv(G))
rf = random_forest(G,q_varying)
nroots(rf)
next(rf) #who points to whom in the forest
````

TODO:
- Add a rng parameter for reproducibility
- Code duplication for random_forest(G::AbstractGraph, q::AbstractFloat/AbstractVector), maybe define a function computing the acceptance to sink node
"""
function random_forest(G::AbstractGraph, q::AbstractFloat)
    n = nv(G)
    root = zeros(Int64, n)
    roots, nroots = Set{Int64}(), 0

    in_tree = falses(n)
    next = zeros(Int64, n)

    d = degree(G)

    @inbounds for i in 1:n
        u = i
        while !in_tree[u]
            if rand() * (q + d[u]) < q
                in_tree[u] = true
                push!(roots,u)
                nroots += 1
                root[u] = u
                next[u] = 0
            else
                next[u] = random_successor(G, u)
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

    return RandomForest(next, roots, nroots, root)
end

function random_forest(G::AbstractGraph, q::AbstractVector)
    n = nv(G)
    @assert length(q) == n

    root = zeros(Int64, n)
    roots, nroots = Set{Int64}(), 0

    in_tree = falses(n)
    next = zeros(Int64, n)

    @inbounds for i in 1:n
        u = i
        while !in_tree[u]
            if rand() * (q[u] + degree(G, u)) < q[u]
                in_tree[u] = true
                push!(roots,u)
                nroots += 1
                root[u] = u
                next[u] = 0
            else
                next[u] = random_successor(G, u)
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
    RandomForest(next, roots, nroots, root)
end

"""
TODO: sum_by function is commented thus not in scope...
"""
function avg_rf(root::Array{Int64,1}, y::Array{Float64,1})
    #    ysum = weighted_sum_by(y,deg,state.root)
    ysum = sum_by(y, root)
    return [ysum[r] for r in root]
end

function sure(y, xhat, nroots, s2)
    err = sum((y .- xhat).^2)
    @show err
    return -length(y)*s2 + err + 2*s2*nroots
end

random_successor(g::SimpleGraph{Int}, i::T) where T<:Int64 = rand(neighbors(g, i))

function random_successor(g::SimpleWeightedGraph, i::T) where T<:Int64
    W = weights(g)
    rn = W.colptr[i]:(W.colptr[i+1]-1)
    w = W.nzval[rn]
    w ./= sum(w)
    u = rand()
    j = 0
    s = 0
    while s < u && j < length(w)
        s += w[j+1]
        j +=1
    end
    W.rowval[rn[1]+j-1]
end

random_successor(g::PreprocessedWeightedGraph, i::T) where T<:Int64 = alias_draw(g, i)

"""
    SimpleDiGraph(rf::RandomForest)

Convert a RandomForest rf to a SimpleDiGraph.

# Example

```
g = grid([3,3])
rf = random_forest(g,.4)
f = SimpleDiGraph(rf)
connected_components(f)
```
"""
function SimpleDiGraph(rf::RandomForest)
    n = length(rf.next)
    ff = SimpleDiGraph(n)
    for i in 1:n
        (rf.next[i] > 0) && add_edge!(ff, i, rf.next[i])
    end
    return ff
end

include("random_spanning_tree.jl")
include("smoothing.jl")
include("moments.jl")
include("multiscale.jl")

# function smooth_rf(G::SimpleGraph{T},q,y::Vector;nrep=10,variant=1) where T
#     xhat = zeros(Float64,length(y));
#     nr = 0;
#     for indr in Base.OneTo(nrep)
#         rf = random_forest(G,q)
#         nr += rf.nroots
#         if variant==1
#             xhat += y[rf.root]
#         elseif variant==2
#             xhat += avg_rf(rf.root,y)
#         end
#     end
#     (est=xhat ./ nrep,nroots=nr/nrep)
# end

# function sum_by(v::Array{T,1}, g::Array{Int64,1}) where T
#     cc = spzeros(Int64,length(v))
#     vv = spzeros(Float64,length(v))
#     for i in 1:length(v)
#         vv[g[i]] += v[i]
#         cc[g[i]] += 1
#     end
#     nz = findnz(vv)
#     for i in nz[1]
#         vv[i] /= cc[i]
#     end
#     vv
# end

# function sum_by(v::Array{T,2}, g::Array{Int64,1}) where T
#     cc = spzeros(Int64,length(v))
#     vv = spzeros(Float64,size(v,1),size(v,2))
#     for i in 1:size(v,1)
# #        @show size(vv[g[i],:]),size(v[i,:])
#         vv[g[i],:] += v[i,:]
#         cc[g[i]] += 1
#     end
#     nz = findnz(cc)
#     for i in nz[1]
#         vv[i,:] /= cc[i]
#     end
#     vv
# end

end # module
