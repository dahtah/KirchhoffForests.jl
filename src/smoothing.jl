struct Partition
    part :: Array{Int,1}
    nparts :: Int
end

function Partition(rf :: RandomForest )
    Partition(Int.(denserank(rf.root)),rf.nroots)
end

function nv(p::Partition)
    length(p.part)
end

function show(io::IO, p::Partition)
    println(io, "Graph partition. Size of original graph $(nv(p)).")
    println(io,"Number of parts $(p.nparts)")
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



function smooth(G :: AbstractGraph{T},q,Y ) where T
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

function smooth_rf(g :: AbstractGraph,q,Y;nrep=10,variant=1,mean_correction=false)
    xhat = zeros(size(Y));
    nr = 0;
    Ym = 0;
    for indr in Base.OneTo(nrep)
        rf = random_forest(g,q)
        nr += rf.nroots
        if variant==1
            xhat += rf*Y
#            xhat += Y[rf.root,:]
        elseif variant==2
            xhat += Partition(rf)*Y
        end
    end
    xhat /= nrep
    if (mean_correction)
        Ym = mean(Y,dims=1)
        Y = Y .- Ym
        xhat = xhat .- mean(xhat,dims=1) .+ Ym
    end
    (est=xhat,nroots=nr/nrep)
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
    #Step 2: assign average to each index according to partition
    X = Matrix{Float64}(undef,n,m)
    @inbounds for i = 1:n
        r = p.part[i]
        @inbounds for j=1:m
            X[i,j] = Z[r,j]
        end
    end
    X
end

function Base.:*(p::Partition,y :: Vector)
    p*reshape(y,:,1)
end




# function smooth_rf_pcg(G :: SimpleGraph{T},q,y :: Vector;nrep=10,alpha=.5,step="fixed",nsteps=10) where T
#     nr = 0;
#     rf = [random_forest(G,q) for _ in 1:nrep]
#     xt = y
#     L = laplacian_matrix(G)
#     cfun = (x) -> (q/2)*sum((x .- y).^2) + .5*x'*L*x
#     gamma =alpha
#     for indr in 1:nsteps
#         gr = q*xt + L*xt - q*y
#         dir = mean([avg_rf(r.root,gr) for r in rf])
#         #dir = (q*U)
#         if (step == "optimal")
#             u = A*dir
#             gamma = ( sum(xhat .* (A*u)) - dot(y,u)   )/dot(u,u)
#             @show gamma
#         elseif (step=="backtrack")
#             curr = cfun(xt)
#             gamma = alpha
# #            while (norm(A*(xhat-gamma*dir) - y) > curr)
#             while (cfun(xt-gamma*dir) > curr)
#                 gamma = gamma/2
#             end
#             @show gamma
#         end
#         xt -= gamma*dir
#         @show cfun(xt)
#     end
#     (est=xt,cost = cfun(xt))
# end


# function smooth_rf_adapt(G :: SimpleGraph{T},q,y :: Vector;nrep=10,alpha=.5,step="fixed") where T
#     nr = 0;
#     rf = random_forest(G,q)
#     xhat = avg_rf(rf.root,y)
# #    @show xhat
#     L = laplacian_matrix(G)
#     cfun = (x) -> (q/2)*sum((x .- y).^2) + .5*x'*L*x
#     A = (L+q*I)/q
#     res = A*xhat - y
#     gamma =alpha
# #    @show res

#     for indr in 2:nrep
#         rf = random_forest(G,q)
#         dir = avg_rf(rf.root,res)
#         if (step == "optimal")
#             u = A*dir
#             gamma = ( sum(xhat .* (A*u)) - dot(y,u)   )/dot(u,u)
#             @show gamma
#         elseif (step=="backtrack")
#             curr = cfun(xhat)
#             gamma = alpha
# #            while (norm(A*(xhat-gamma*dir) - y) > curr)
#             while (cfun(xhat-gamma*dir) > curr)
#                 gamma = gamma/2
#             end
#             @show gamma
#         end

#         xhat -= gamma*dir
#         res = A*xhat - y
#         @show cfun(xhat)
#         nr += rf.nroots
#     end
#     (est=xhat,nroots=nr/nrep)
# end

