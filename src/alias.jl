
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
    W = Matrix(weights(g))
    n = size(g)[1]
    K = spzeros(Int64,n,n)
    U = spzeros(Float64,n,n)

    for k = 1: n
        probs = W[k,:]
        # rn = W.colptr[k]:(W.colptr[k+1]-1)
        # probs = W.nzval[rn]
        probs /= sum(probs)
        # Overfull and Underfull stacks
        ofull = Int64[]
        ufull = Int64[]
        # For keeping indices

        # For keeping probabilities
        # Initialize indices and stacks with original probabilities
        for i = 1 : n
            U[k,i] = n*probs[i]
            if(U[k,i] > 1)
                push!(ofull,i)
            elseif(U[k,i] < 1)
                push!(ufull,i)
            else
                K[k,i] = i
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
    end
    K,U
end
function alias_draw(g,i)
    """
    Drawing procedure of alias method after the preprocessing
    Its cost is constant!
    """
    v = rand(1:size(g.P,1))
    if(rand() < g.P[i,v])
        sample = v
    else
        sample = g.K[i,v]
    end
    Int64(sample)
end
