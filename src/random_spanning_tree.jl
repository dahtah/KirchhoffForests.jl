"""
    random_spanning_tree(g,[r])

Generate a (uniform) random spanning tree using Wilson's algorithm.
A spanning tree of g is a connected subgraph of g that's cycle-free, i.e. a tree that includes only edges from g and connects every node.
If you specify the root of the tree, the function produces a spanning tree that is picked uniformly among all trees rooted at r.
If you do not specify a root, the function produces a random tree from g, picking *uniformly* among all spanning trees of g (over all possible roots).

NB: a graph must be connected in order to have a spanning tree. By default, the function checks that g is connected (in order to avoid an infinite loop). If you are positive g is connected, use force=true.

### Arguments
- g: a graph
- optional: r, index of a node to serve as the root
- force: if true, skip the connectivity test

### Output
If the root is specified, returns a tree, represented as a SimpleDiGraph. If it isn't, returns a named tuple with "tree": the tree and "root": the root.


### Examples

```@example
julia> g = cycle_graph(4)
{4, 4} undirected simple Int64 graph

julia> random_spanning_tree(g).tree |> edges |> collect
3-element Array{Graphs.SimpleGraphs.SimpleEdge{Int64},1}:
 Edge 2 => 3
 Edge 3 => 4
 Edge 4 => 1
```
"""
function random_spanning_tree(g :: SimpleGraph{T}, r :: Integer; force=false) where T
    r in vertices(g) || throw(BoundsError("Root r must be one of the vertices"))
    if (!force)
        is_connected(g) || throw(ArgumentError("Graph must be connected"))
    end

    n = nv(g)
    in_tree = falses(n)
    next = zeros(T,n)
    in_tree[r] = true
    # we follow closely Wilson's extremely elegant pseudo-code
    for i in vertices(g)
        u = i
        while !in_tree[u] #run a loop-erased random walk
            nn = outneighbors(g, u)
            length(nn) == 0 && throw(ArgumentError("No spanning tree with this root exists"))
            next[u]= rand(nn)
            u = next[u]
        end
        #Retrace steps, erasing loops
        u = i
        while !in_tree[u]
            in_tree[u] = true
            u = next[u]
        end
    end
    return tree_from_next(next,vertices(g))
end

#nb: roots are uniformly distributed in an undirected graph
function random_spanning_tree(g :: SimpleGraph{T}; force=false) where T
    root = rand(vertices(g))
    tree = random_spanning_tree(g,root;force=force)
    (root=root,tree=tree)
end

# Given a vector of pointers "next", make a tree of DiGraph type
function tree_from_next(next :: Array{T,1}, nodes :: AbstractArray{T,1}) where T
    tree = SimpleDiGraph{T}(length(next))
    for v in nodes
        next[v] > 0 && add_edge!(tree,v,next[v])
    end
    tree
end
