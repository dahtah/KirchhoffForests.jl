
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
    D = (sum(W.>0,dims=1))[1,:]
    K = spzeros(Int64,n,n)
    U = spzeros(Float64,n,n)

    for k = 1: n

        nhbrs = neighbors(g,k)
        nhbrssize = size(nhbrs,1)

        probs = zeros(Float64,nhbrssize)

        Kvec = zeros(Int64,nhbrssize)
        Uvec = zeros(Float64,nhbrssize)

        rn = W.colptr[k]:(W.colptr[k+1]-1)
        probs = W.nzval[rn] ## <---- Lookup time for sparse array!!!
        probs /= sum(probs)
        # Overfull and Underfull stacks
        ofull = Int64[]
        ufull = Int64[]
        # For keeping indices

        # For keeping probabilities
        # Initialize indices and stacks with original probabilities
        for (idx, i) in enumerate(probs)
            Uvec[idx] = nhbrssize*i
            if(Uvec[idx] > 1)
                push!(ofull,idx)
            elseif(Uvec[idx] < 1)
                push!(ufull,idx)
            else
                Kvec[idx] = idx
            end

        end
        # Loop until all bins are "equally full"
        while !(isempty(ofull) && isempty(ufull))
            i = pop!(ofull)
            j = pop!(ufull)
            Kvec[j] = i
            # Recompute overfull bin
            Uvec[i] = Uvec[i] + Uvec[j] - 1

            # Reassign the bin
            if(Uvec[i] - 1.0 > 0.000001) # Due to floating point errors (but not elegant)
                push!(ofull,i)
            elseif((Uvec[i] - 1.0 < -0.000001))
                push!(ufull,i)
            else
                Kvec[i] = i
            end
        end
        K[nhbrs,k] .= nhbrs[Kvec]
        U[nhbrs,k] .= Uvec
    end
    K,U
end
function alias_draw(g,i)
    """
    Drawing procedure of alias method after the preprocessing
    Its cost is constant!
    """
    rn = g.P.colptr[i]:(g.P.colptr[i+1]-1)
    v = rand(rn)

    if(rand() < g.P.nzval[v])
        return g.P.rowval[v]
    else
        return g.K.nzval[v]
    end
end
