module RandomForests
using LightGraphs,LinearAlgebra,SparseArrays,SimpleWeightedGraphs
import StatsBase.denserank,Statistics.mean,Base.show,LightGraphs.SimpleDiGraph
export random_forest,smooth,smooth_rf,smooth_rf_adapt,RandomForest,SimpleDiGraph

struct RandomForest
    next :: Array{Int}
    roots :: Set{Int}
    nroots :: Int
    root :: Array{Int,1}
end

function show(io::IO, rf::RandomForest)
    println(io, "Random forest. Size of original graph $(length(rf.next)).")
    println(io,"Number of trees $(rf.nroots)")
end


"""
    random_forest(G::AbstractGraph,q)

Run Wilson's algorithm on G to generate a random forest with parameter "q". q determines the probability that
the random walk is interrupted at a node. If q is a scalar, that probability equals q/(q+d[i]) at node i with
degree d[i]. If q is a vector, it equals q[i]/(q[i]+d[i]).

# Example

'''
using LightGraphs
G = grid([3,3])
random_forest(G,.4)
q_varying = rand(nv(G))
random_forest(G,q_varying)
'''

# Warning
The value of next[i] for i root is set to 0, because roots do not have a successor. 
"""
function random_forest(G::AbstractGraph,q::AbstractFloat)
    roots = Set{Int64}()
    root = zeros(Int64,nv(G))
    nroots = Int(0)

    n = nv(G)
    in_tree = falses(n)
    next = zeros(Int64,n)
    @inbounds for i in 1:n
        u = i
        
        while !in_tree[u]
            if (rand() < q/(q+degree(G,u)))
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

function random_successor(G::SimpleGraph{T},i :: T) where T <: Int
    nbrs = neighbors(G, i)
    rand(nbrs)
end

function random_successor(g :: SimpleWeightedGraph,i :: T) where T <: Int
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

function SimpleDiGraph(rf :: RandomForest)
    n = length(rf.next)
    ff = SimpleDiGraph(n)
    for i in 1:n
        (rf.next[i] > 0) && add_edge!(ff,i,rf.next[i])
    end
    ff
end


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



function sum_by(v :: Array{T,1}, g :: Array{Int64,1}) where T
    cc = spzeros(Int64,length(v))
    vv = spzeros(Float64,length(v))
    for i in 1:length(v)
        vv[g[i]] += v[i]
        cc[g[i]] += 1
    end
    nz = findnz(vv)
    for i in nz[1]
        vv[i] /= cc[i]
    end
    vv
end

function sum_by(v :: Array{T,2}, g :: Array{Int64,1}) where T
    cc = spzeros(Int64,length(v))
    vv = spzeros(Float64,size(v,1),size(v,2))
    for i in 1:size(v,1)
#        @show size(vv[g[i],:]),size(v[i,:])
        vv[g[i],:] += v[i,:]
        cc[g[i]] += 1
    end
    nz = findnz(cc)
    for i in nz[1]
        vv[i,:] /= cc[i]
    end
    vv
end




end # module

