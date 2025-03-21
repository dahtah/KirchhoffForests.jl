#Basic support for Kirchoff Forests

struct KirchhoffForest
    next::Array{Int}  # todo what size?
    roots::Set{Int}
    nroots::Int
    root::Array{Int,1}
end

function show(io::IO, rf::KirchhoffForest)
    println(io, "Random forest. Size of original graph $(nv(rf)).")
    println(io, "Number of trees $(nroots(rf))")
end

nv(rf::KirchhoffForest) = length(rf.next)
nroots(rf::KirchhoffForest) = rf.nroots
ne(rf::KirchhoffForest) = nv(rf) - nroots(rf)

"""
    next(rf::KirchhoffForest)

Return a vector of indices v, where v[i] = j means that node i points to node j
in the forest. If v[i] = 0 i is a root.
"""
next(rf::KirchhoffForest) = rf.next

outneighbors(rf::KirchhoffForest,i) = rf.next[i] > 0 ? [rf.next[i]] : Array{Int64,1}()

"""
    random_forest(G::AbstractGraph,q)

Run Wilson's algorithm on G to generate a random forest with parameter "q". q determines the probability that
the random walk is interrupted at a node. If q is a scalar, that probability equals q/(q+d[i]) at node i with
degree d[i]. If q is a vector, it equals q[i]/(q[i]+d[i]).

# Example

```
using Graphs
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
function random_forest(G::AbstractGraph, q)
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

    return KirchhoffForest(next, roots, nroots, root)
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
    KirchhoffForest(next, roots, nroots, root)
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


"""
    SimpleDiGraph(rf::KirchhoffForest)

Convert a KirchhoffForest rf to a SimpleDiGraph.

# Example

```
g = grid([3,3])
rf = random_forest(g,.4)
f = SimpleDiGraph(rf)
connected_components(f)
```
"""
function SimpleDiGraph(rf::KirchhoffForest)
    n = length(rf.next)
    ff = SimpleDiGraph(n)
    for i in 1:n
        (rf.next[i] > 0) && add_edge!(ff, i, rf.next[i])
    end
    return ff
end
