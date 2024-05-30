using Random
function random_forest(G::AbstractGraph,q::Number,rng::AbstractRNG)
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
            if (((q+d[u]))*rand(rng) < q)
                in_tree[u] = true
                push!(roots,u)
                nroots+=1
                root[u] = u
                next[u] = 0
            else
                next[u] = random_successor(G,u,rng)
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
    KirchoffForest(next,roots,nroots,root)
end

function random_forest(G::AbstractGraph,q::Number,B::AbstractVector,rng::AbstractRNG)
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
            if ((((q+d[u]))*rand(rng) < q) || u in B)
                in_tree[u] = true
                push!(roots,u)
                nroots+=1
                root[u] = u
                next[u] = 0
            else
                next[u] = random_successor(G,u,rng)
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
    KirchoffForest(next,roots,nroots,root)
end


function random_forest(G::AbstractGraph,q::AbstractVector,rng::AbstractRNG)
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
            if (rand(rng) < q[u]/(q[u]+degree(G,u)))
                in_tree[u] = true
                push!(roots,u)
                nroots+=1
                root[u] = u
                next[u] = 0
            else
                next[u] = random_successor(G,u,rng)
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
    KirchoffForest(next,roots,nroots,root)
end

function random_forest(G::AbstractGraph,q::AbstractVector,B::AbstractVector,rng::AbstractRNG)
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
            if ((rand(rng) < q[u]/(q[u]+degree(G,u))) || (u in B))
                in_tree[u] = true
                push!(roots,u)
                nroots+=1
                root[u] = u
                next[u] = 0
            else
                next[u] = random_successor(G,u,rng)
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
    KirchoffForest(next,roots,nroots,root)
end


function random_successor(G::AbstractGraph,i :: T,rng::AbstractRNG) where T <: Int64
    nbrs = neighbors(G, i)
    rand(rng,nbrs)
end

function random_successor(g :: SimpleWeightedGraph,i :: T,rng::AbstractRNG) where T <: Int64
    W = weights(g)
    rn = W.colptr[i]:(W.colptr[i+1]-1)
    w = W.nzval[rn]
    w /= sum(w)
    u = rand(rng)
    j = 0
    s = 0
    while s < u && j < length(w)
        s+= w[j+1]
        j+=1
    end
    W.rowval[rn[1]+j-1]
end

function random_successor(g :: PreprocessedWeightedGraph,i :: T,rng::AbstractRNG) where T <: Int64

    sample = alias_draw(g,i,rng)
    sample
end
