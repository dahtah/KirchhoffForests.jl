#Partition, as implemented here, is not efficient!
#Should update



struct Partition
    part :: Array{Int,1}
    nparts :: Int
    sizep :: Array{Int,1}
end

function Partition(rf :: RandomForest )
    part=Int.(denserank(rf.root))
    sizep = counts(part)
    Partition(part,rf.nroots,sizep)
end

function nv(p::Partition)
    length(p.part)
end

function show(io::IO, p::Partition)
    println(io, "Graph partition. Size of original graph $(nv(p)).")
    println(io,"Number of parts $(p.nparts)")
end

function laplacian_matrix(g :: PreprocessedWeightedGraph)
    W = g.weights
    s = sum(W,dims=1)
    D = spdiagm(0 => s[:])
    D - W
end

function laplacian_matrix(g :: SimpleWeightedGraph)
    W = weights(g)
    s = sum(W,dims=1)
    D = spdiagm(0 => s[:])
    D - W
end

function laplacian_matrix(g :: SimpleGraph)
    A = adjacency_matrix(g)
    s = degree(g)
    D = spdiagm(0 => s[:])
    return D - A
end



#average the signals in Y over a partition
function avg_rf(roots,Y)
    n = length(roots)
    m = size(Y,2)
    #denserank labels the roots vector from 1...nroots
    rk = Int.(denserank(roots))
    nroots = maximum(rk)
    Z = zeros(nroots,m)
    ns = zeros(Int,nroots)
    @inbounds for i = 1:n
        r = rk[i]
        nc = ns[r]+1
        @inbounds for j = 1:m
            Z[r,j] *= ((nc-1)/nc)
            Z[r,j] += Y[i,j]/nc
        end
        ns[r] = nc
    end
    X = Matrix{Float64}(undef,n,m)
    @inbounds for i = 1:n
        r = rk[i]
        @inbounds for j=1:m
            X[i,j] = Z[r,j]
        end
    end
    X
end



function smooth(G :: AbstractGraph{T},q :: Float64,Y ) where T
    L=laplacian_matrix(G)
    q*((L+q*I)\Y)
end

@doc raw"""
   smooth(g :: AbstractGraph{T},q,Y )

Smooth signal over graph. Given a vector ``\mathbf{y}`` of size nv(g), compute
``q(q\mathbf{I}+\mathbf{L})^{-1}\mathbf{y}``, where ``\mathbf{L}`` is the graph
Laplacian and q > 0 is a regularisation coefficient (the smaller q, the stronger
the smoothing).

If Y is a matrix then this function computes
``q(q\mathbf{I}+\mathbf{L})^{-1}\mathbf{Y}``. The linear system is solved using
a direct method.

# Example

```
g = grid([10])
t = LinRange(0,1,10)
y = sin.(6*pi*t)
smooth(g,.1,y)
smooth(g,10.1,y)
```

"""
function smooth(g :: AbstractGraph{T},q :: Vector,Y ) where T
    L=laplacian_matrix(g)
    Q = diagm(0=>q)
    (L+Q)\(Q*Y)
end

function smooth(g :: AbstractGraph{T},q :: Float64,Y :: SparseMatrixCSC) where T
    L=laplacian_matrix(g)
    C = cholesky(L+q*I)
    q*(C\Y)
end

function smooth(g :: AbstractGraph{T},q :: Vector,Y :: SparseMatrixCSC) where T
    L=laplacian_matrix(g)
    Q = diagm(0=>q)
    C = cholesky(L+Q)
    C\(Q*Y)
end


function smooth_rf(g :: AbstractGraph,q :: Float64,Y,rootset=[];nrep=10,variant=1,mean_correction=false)
    xhat = zeros(size(Y));
    nr = 0;
    Ym = 0;
    diagK = zeros(nv(g))

    for indr in Base.OneTo(nrep)
        rf = random_forest(g,q,rootset)
        nr += rf.nroots
        roots =[i for i in rf.roots]
        if variant==1
            xhat += rf*Y
            diagK[roots] .+= 1
        elseif variant==2
            xhat += Partition(rf)*Y
            diagK += diag(Partition(rf))
        end
    end
    xhat /= nrep
    if (mean_correction)
        Ym = mean(Y,dims=1)
        Y = Y .- Ym
        xhat = xhat .- mean(xhat,dims=1) .+ Ym
    end
    (est=xhat,nroots=nr/nrep, diagK = diagK/nrep)
end

function smooth_rf(g :: AbstractGraph,q :: Vector,Y,rootset=[];nrep=10,variant=1,mean_correction=false)
    xhat = zeros(size(Y));
    nr = 0;
    Ym = 0;
    Yq = Diagonal(q)*Y
    diagK = zeros(nv(g))

    for indr in Base.OneTo(nrep)
        rf = random_forest(g,q,rootset)
        nr += rf.nroots
        roots =[i for i in rf.roots]
        if variant==1
            xhat += rf*Y
            diagK[roots] .+= 1
        elseif variant==2
            m = Partition(rf)*q
            xhat += Partition(rf)*Yq ./ m
            diagK +=  (diag(Partition(rf)) .* q) ./ m
        end
    end
    xhat /= nrep
    if (mean_correction)
        Ym = mean(Y,dims=1)
        Y = Y .- Ym
        xhat = xhat .- mean(xhat,dims=1) .+ Ym
    end
    (est=xhat,nroots=nr/nrep,diagK=diagK/nrep)
end



"""
*(rf::RandomForest,Y :: Matrix)

Treating the random forest as a linear operator, propagate the value of y at
the root to the rest of the tree.

# Example

```
g = grid([5])
rf = random_forest(g,.5)
rf*collect(1:nv(g))
```

"""
function Base.:*(rf::RandomForest, Y::Matrix)
    Y[rf.root,:]
end

function Base.:*(rf::RandomForest, Y::Vector)
    Y[rf.root]
end


"""
*(p::Partition,Y :: Matrix)

Treating the graph partition as a linear operator, compute the average of Y
over the partition.

# Example

```
g = grid([5])
rf = random_forest(g,.5)
p = Partition(rf)
p*collect(1:nv(g))
```

"""
function Base.:*(p::Partition,Y :: Matrix)
    #Step 1: compute average of Y in each subset
    propagate(p,average(p,Y))
end

function Base.:*(p::Partition,y :: Vector)
    p*reshape(y,:,1) |> vec
end

function LinearAlgebra.:diag(p::Partition)
    w = 1 ./ p.sizep
    w[p.part]
end


function average(p::Partition,y :: Real)
    repeat([y],p.nparts)
end


function average(p::Partition,y :: Vector)
    average(p,reshape(y,:,1)) |> vec
end

function propagate(p::Partition,Z :: Matrix)
    m = size(Z,2)
    X = Matrix{Float64}(undef,nv(p),m)
    @inbounds for i = 1:nv(p)
        r = p.part[i]
        @inbounds for j=1:m
            X[i,j] = Z[r,j]
        end
    end
    X
end


function grad_descent(g::SimpleWeightedGraph,y,x0,q;α=.1)
    L = laplacian_matrix(g)
    gr = L*x0 .+ q.*(x0 .- y)
    x = Vector(x0 .- α*(gr)./diag(L))
    (x = x,res= Vector(x .- y + L*x ./ q) )
end

function propagate(p::Partition,z :: Vector)
    propagate(p,reshape(z,:,1)) |> vec
end

function average(p::Partition,Y :: Matrix)
    n = length(p.part)
    m = size(Y,2)

    Z = zeros(p.nparts,m)
    ns = zeros(Int,p.nparts)
    #Step 1: compute average of Y in each subset
    @inbounds for i = 1:n
        r = p.part[i]
        nc = ns[r]+1
        @inbounds for j = 1:m
            Z[r,j] *= ((nc-1)/nc)
            Z[r,j] += Y[i,j]/nc
        end
        ns[r] = nc
    end
    Z
end

#sum Y in each subset
function sum(p::Partition,Y :: Matrix)
    n = length(p.part)
    m = size(Y,2)

    Z = zeros(p.nparts,m)
    @inbounds for i = 1:n
        r = p.part[i]
        @inbounds for j = 1:m
            Z[r,j] += Y[i,j]
        end
    end
    Z
end

function sum(p::Partition,y :: Vector)
    sum(p,reshape(y,:,1)) |> vec
end
