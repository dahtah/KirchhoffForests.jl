"""
    random_spanning_tree(g, [r])

Generate a (uniform) random spanning tree (https://en.wikipedia.org/wiki/Spanning_tree) using Wilson's algorithm (https://dl.acm.org/doi/10.1145/237814.237880).
A spanning tree of connected graph g is a connected subgraph of g that's cycle-free, i.e. a tree that includes only edges from g and connects every node.
If you specify the root of the tree, the function produces a spanning tree that is picked uniformly among all trees rooted at r.
If you do not specify a root, the function produces a random tree from g, picking *uniformly* among all spanning trees of g (over all possible roots).

NB: a graph must be connected in order to have a spanning tree. By default, the function checks that g is connected (in order to avoid an infinite loop). If you are positive g is connected, use force=true.

### Arguments
- g: a graph
- optional: r, index of a node to serve as root
- force: if true, skip connectivity test

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
TODO: Maybe consider the function random_spanning_tree as a meta function with a switch wilson / aldous..
"""
function random_spanning_tree(g::SimpleGraph{T}, r::Integer; force=false) where T

    r in vertices(g) || throw(BoundsError("Root r must be one of the vertices"))
    if !force
        is_connected(g) || throw(ArgumentError("Graph must be connected"))
    end

    n = nv(g)
    in_tree = falses(n)
    next = zeros(T, n)

    in_tree[r] = true

    # Natural loop-erased random walk
    for v in vertices(g)
        # Natural walk on the graph until meet a "in_tree" vertex
        current_v = v
        while !in_tree[current_v]
            nghbrs = outneighbors(g, current_v)
            next[current_v] = rand(rng, nghbrs)  # empty handled by rand
            current_v = next[current_v]
        end
        # Retrace steps, erase loops, keep only path v -> first loopy vertex
        current_v = v
        while !in_tree[current_v]
            in_tree[current_v] = true
            current_v = next[current_v]
        end
    end
    return tree_from_next(next, vertices(g))
end

#nb: roots are uniformly distributed in an undirected graph
function random_spanning_tree(g::SimpleGraph{T}; force=false) where T
    root = rand(vertices(g))
    tree = random_spanning_tree(g, root; force=force)
    return (root=root, tree=tree)
end

# Given a vector of pointers "next", make a tree of DiGraph type
function tree_from_next(next::Array{T,1}, nodes::AbstractArray{T,1}) where T
    tree = SimpleDiGraph{T}(length(next))
    for v in nodes
        next[v] > 0 && add_edge!(tree, v, next[v])
    end
    return tree
end
