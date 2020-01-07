module RandomForests
using LightGraphs,LinearAlgebra,SparseArrays,SimpleWeightedGraphs
import StatsBase.denserank,Statistics.mean,Base.show,Base.sum,
StatsBase.counts
import LightGraphs.SimpleDiGraph,LightGraphs.nv,LightGraphs.ne,LightGraphs.degree,LightGraphs.outneighbors
import SimpleWeightedGraphs.AbstractSimpleWeightedGraph
import LightGraphs:
nv,ne,outneighbors,is_directed,inneighbors

include("./alias.jl")

export random_forest,smooth,smooth_rf,smooth_rf_adapt,RandomForest,
      SimpleDiGraph,nroots,next,Partition,PreprocessedWeightedGraph
export reduced_graph,smooth_ms
export self_roots
export random_successor

struct RandomForest
    next :: Array{Int}
    roots :: Set{Int}
    nroots :: Int
    root :: Array{Int,1}
end

# abstract type AbstractSimpleWeightedGraph{T<:Integer,U<:Real} <: AbstractGraph{T} end
struct PreprocessedWeightedGraph{T<:Integer, U<:Real} <: AbstractSimpleWeightedGraph{T, U}
    weights::SparseMatrixCSC{U, T}
    K :: Matrix{Int64}
    P :: Matrix{Float64}
end

function PreprocessedWeightedGraph(adjmx::SparseMatrixCSC{U,T}) where T <: Integer where U <: Real
    K,P = alias_preprocess(SimpleWeightedGraph(adjmx))
    PreprocessedWeightedGraph{T, U}(adjmx,K,P)
end
PreprocessedWeightedGraph(g::LightGraphs.SimpleGraphs.SimpleGraph{T}, ::Type{U}=Float64) where T <: Integer where U <: Real =
    PreprocessedWeightedGraph(adjacency_matrix(g, U))
PreprocessedWeightedGraph(g::SimpleWeightedGraph)= PreprocessedWeightedGraph(g.weights)
inneighbors(g::PreprocessedWeightedGraph, v::Integer) = g.weights[v,:].nzind
is_directed(::Type{PreprocessedWeightedGraph}) = false
is_directed(::Type{PreprocessedWeightedGraph{T, U}}) where T where U = false
is_directed(g::PreprocessedWeightedGraph) = false
ne(g::PreprocessedWeightedGraph) = nnz(g.weights)/2
degree(g::PreprocessedWeightedGraph) = sum(g.weights,dims=1)
degree(g::SimpleWeightedGraph) = sum(g.weights,dims=1)


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
    next(rf::RandomForest)

Return a vector of indices v, where v[i] = j means that node i points to node j
in the forest. If v[i] = 0 i is a root.
"""
function next(rf::RandomForest)
    rf.next
end

function outneighbors(rf::RandomForest,i)
    rf.next[i] > 0 ? [rf.next[i]] : Array{Int64,1}()
end

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

"""
function random_forest(G::AbstractGraph,q::AbstractFloat)
    roots = Set{Int64}()
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

function random_forest(G::AbstractGraph,q::AbstractVector)
    @assert length(q)==nv(G)
    roots = Set{Int64}()
    root = zeros(Int64,nv(G))
    nroots = Int(0)

    n = nv(G)

    in_tree = falses(n)
    next = zeros(Int64,n)
    @inbounds for i in 1:n
        u = i

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
    (next=next,roots=roots,nroots=nroots,root=root)
end

function avg_rf(root :: Array{Int64,1},y :: Array{Float64,1})
    xhat = zeros(Float64,length(y))
    #    ysum = weighted_sum_by(y,deg,state.root)
    ysum = sum_by(y,root)
    for v in 1:length(xhat)
        xhat[v] = ysum[root[v]]
    end
    xhat
end

function sure(y,xhat,nroots,s2)
    err = sum((y .- xhat).^2)
    @show err
    -length(y)*s2+err+2*s2*nroots
end

function random_successor(G::SimpleGraph{Int},i :: T) where T <: Int64
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

```
g = grid([3,3])
rf = random_forest(g,.4)
f = SimpleDiGraph(rf)
connected_components(f)
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

include("smoothing.jl")
include("moments.jl")
include("multiscale.jl")

# function smooth_rf(G :: SimpleGraph{T},q,y :: Vector;nrep=10,variant=1) where T
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



# function sum_by(v :: Array{T,1}, g :: Array{Int64,1}) where T
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

# function sum_by(v :: Array{T,2}, g :: Array{Int64,1}) where T
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
