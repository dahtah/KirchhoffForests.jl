using Graphs,LinearAlgebra,SparseArrays,SimpleWeightedGraphs

export alias_preprocess
export alias_draw

struct PreprocessedWeightedGraph{T<:Integer,U<:Real}<:AbstractSimpleWeightedGraph{T,U}
    weights::SparseMatrixCSC{U,T}
    K::SparseMatrixCSC{Int64,T}
    P::SparseMatrixCSC{Float64,T}
end

function PreprocessedWeightedGraph(adjmx::SparseMatrixCSC{U,T}) where {T<:Integer,U<:Real}
    K, P = alias_preprocess(SimpleWeightedGraph(adjmx))
    PreprocessedWeightedGraph{T, U}(adjmx,K,P)
end

PreprocessedWeightedGraph(g::SimpleWeightedGraph) = PreprocessedWeightedGraph(g.weights)
PreprocessedWeightedGraph(g::SimpleGraph{T}, ::Type{U}=Float64) where T <: Integer where U <: Real =
    PreprocessedWeightedGraph(adjacency_matrix(g, U))

inneighbors(g::PreprocessedWeightedGraph, v::Integer) = g.weights[v,:].nzind
is_directed(::Type{PreprocessedWeightedGraph}) = false
is_directed(::Type{PreprocessedWeightedGraph{T, U}}) where {T, U} = false
is_directed(g::PreprocessedWeightedGraph) = false
##FIXME: ne should return an integer, and this leads to incorrect results
##if there are self-edges
ne(g::PreprocessedWeightedGraph) = nnz(g.weights)/2
nv(g::PreprocessedWeightedGraph) = size(g.weights,1)
degree(g::PreprocessedWeightedGraph) = sum(g.weights,dims=1)
degree(g::SimpleWeightedGraph) = sum(g.weights,dims=1)

function show(io::IO, g::PreprocessedWeightedGraph)
    println(io, "Preprocessed Weighted Graph with $(nv(g)) vertices and $(ne(g)) edges.")
end
"""
    This is the implementation of the preprocessing for the alias method.
    Overall cost for a graph is O(N^2)
    https://en.wikipedia.org/wiki/Alias_method
"""
function alias_preprocess(g::SimpleWeightedGraph)

    W = weights(g)

    n = size(g)[1]
    K = spzeros(Int64, n, n)
    U = spzeros(Float64, n, n)

    for k in 1:n

        nhbrs = neighbors(g, k)
        nhbrssize = size(nhbrs, 1)

        probs = zeros(Float64, nhbrssize)

        Kvec = zeros(Int64, nhbrssize)
        Uvec = zeros(Float64, nhbrssize)

        rn = W.colptr[k]:(W.colptr[k+1]-1)
        probs = W.nzval[rn]  # <---- Lookup time for sparse array!!!
        probs /= sum(probs)  # TODO ./= possible ?
        # Overfull and Underfull stacks
        ofull = Int64[]
        ufull = Int64[]
        # For keeping indices : TODO, what is this line made for ?

        # For keeping probabilities
        # Initialize indices and stacks with original probabilities
        for (i, prob) in enumerate(probs)
            Uvec[i] = nhbrssize * prob
            if Uvec[i] > 1
                push!(ofull, i)
            elseif Uvec[i] < 1
                push!(ufull, i)
            else  # Uvec[i] == 1
                Kvec[i] = i
            end
        end

        tol = 1e-6  # Due to floating point errors (but not elegant)
        # Loop until all bins are "equally full"
        while !(isempty(ofull) && isempty(ufull))
            i = pop!(ofull)
            j = pop!(ufull)
            Kvec[j] = i
            # Recompute overfull bin
            Uvec[i] += Uvec[j] - 1

            # Reassign the bin
            if Uvec[i] - 1.0 > tol
                push!(ofull, i)
            elseif Uvec[i] - 1.0 < -tol
                push!(ufull, i)
            else
                Kvec[i] = i
            end
        end
        K[nhbrs, k] = nhbrs[Kvec]
        U[nhbrs, k] = Uvec
    end

    return K, U
end
    

"""
Drawing procedure of alias method after the preprocessing
Its cost is constant!
"""
function alias_draw(g::PreprocessedWeightedGraph, i)
    rn = g.P.colptr[i]:(g.P.colptr[i+1]-1)
    v = rand(rn)
<
    return rand() < g.P.nzval[v] ? g.P.rowval[v] : g.K.nzval[v]
end

random_successor(g::PreprocessedWeightedGraph, i) = alias_draw(g, i)
