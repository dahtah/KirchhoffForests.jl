
using LightGraphs,LinearAlgebra,SparseArrays,SimpleWeightedGraphs
using RandomForests

export alias_preprocess
export alias_draw

function alias_preprocess(g :: SimpleWeightedGraph)
"""
    This is the implementation of the preprocessing for the alias method.
    Overall cost for a graph is O(N^2)
    https://en.wikipedia.org/wiki/Alias_method
"""
    W = weights(g)
    n = size(g)[1]
    dmax = maximum(sum(W.>0,dims=1))
    K = zeros(Int64,n,dmax)
    U = zeros(Float64,n,dmax)
    for k = 1: n
        probs = zeros(Float64,dmax)
        nhbrs = neighbors(g,k)
        nhbrssize = size(nhbrs,1)

        probs[1:nhbrssize] = W[k,nhbrs] ## <---- Lookup time for sparse array!!!
        probs /= sum(probs)
        # Overfull and Underfull stacks
        ofull = Int64[]
        ufull = Int64[]
        # For keeping indices

        # For keeping probabilities
        # Initialize indices and stacks with original probabilities
        for (idx, i) in enumerate(probs)
            U[k,idx] = dmax*i
            if(U[k,idx] > 1)
                push!(ofull,idx)
            elseif(U[k,idx] < 1)
                push!(ufull,idx)
            else
                K[k,idx] = idx
            end

        end
        # Loop until all bins are "equally full"
        while !(isempty(ofull) && isempty(ufull))
            i = pop!(ofull)
            j = pop!(ufull)
            K[k,j] = i
            # Recompute overfull bin
            U[k,i] = U[k,i] + U[k,j] - 1

            # Reassign the bin
            if(U[k,i] - 1.0 > 0.000001) # Due to floating point errors (but not elegant)
                push!(ofull,i)
            elseif((U[k,i] - 1.0 < -0.000001))
                push!(ufull,i)
            else
                K[k,i] = i
            end
        end
        # Due to floating point errors
        # if (!isempty(ofull))
        #     U[k,ofull] .= 1.0
        # elseif(!isempty(ufull))
        #     U[k,ufull] .= 1.0
        # end
    end
    K,U
end
function alias_draw(g,i)
    """
    Drawing procedure of alias method after the preprocessing
    Its cost is constant!
    """
    nhbrs = neighbors(g,i)
    n,dmax = size(g.P)
    v = rand(1:dmax)

    if(rand() < g.P[i,v])
        sample = v
    else
        sample = g.K[i,v]
    end
    Int64(nhbrs[sample])
end
