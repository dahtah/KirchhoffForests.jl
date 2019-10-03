module RandomForests
using LightGraphs,LinearAlgebra,SparseArrays
import StatsBase.denserank
export random_forest,smooth,smooth_rf


function random_forest(G::Graph,q::AbstractFloat)
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

function smooth_over_partition(G :: SimpleGraph,root :: Array{Int64,1},y :: Array{Float64,1})
    xhat = zeros(Float64,nv(G))
    #    ysum = weighted_sum_by(y,deg,state.root)
    ysum = sum_by(y,root)
    for v in vertices(G)
        xhat[v] = ysum[root[v]]
    end
    xhat
end

# function smooth_over_partition(G :: SimpleGraph,root :: Array{Int64,1},Y :: Array)
#     xhat = zeros(size(Y))
#     #    ysum = weighted_sum_by(y,deg,state.root)
#     ysum = sum_by(Y,root)
#     for v in vertices(G)
#         xhat[v,:] = ysum[root[v],:]
#     end
#     xhat
# end



function sure(y,xhat,nroots,s2)
    err = sum((y .- xhat).^2)
    @show err
    -length(y)*s2+err+2*s2*nroots
end

function random_successor(G::SimpleGraph{T},i :: T) where T <: Int
    nbrs = neighbors(G, i)
    rand(nbrs)
end


function avg_rf(roots,Y)
    n = length(roots)
    m = size(Y,2)
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

function smooth_wilson_adapt(G :: SimpleGraph{T},q,y :: Vector;nrep=10,alpha=.5,step="fixed") where T
    nr = 0;
    rf = random_forest(G,q)
    xhat = smooth_over_partition(G,rf.root,y)
#    @show xhat
    L = lap(G)
    A = (L+q*I)/q
    res = A*xhat - y
    gamma =alpha
#    @show res
    #    @show norm(res)


    for indr in 2:nrep
        rf = random_forest(G,q)
        dir = smooth_over_partition(G,rf.root,res)
        if (step == "optimal")
            u = A*dir
            gamma = ( xhat'*A*u - dot(y,u)   )/dot(u,u)
 #           @show gamma
        elseif (step=="backtrack")
            curr = norm(res)
            gamma = alpha
            while (norm(A*(xhat-gamma*dir) - y) > curr)
                gamma = gamma/2
            end
#            @show gamma
        end

        xhat -= gamma*dir
        res = A*xhat - y
#        @show res
 #       @show norm(res)
        nr += rf.nroots
    end
    (xhat=xhat,nroots=nr/nrep)
end

function smooth(G :: SimpleGraph{T},q,y :: Vector) where T
    L=laplacian_matrix(G)
    q*((L+q*I)\y)
end

function smooth(G :: SimpleGraph{T},q,Y :: Matrix) where T
    L=laplacian_matrix(G)
    q*((L+q*I)\Y)
end

function smooth(G :: SimpleGraph{T},q,Y :: SparseMatrixCSC) where T
    L=laplacian_matrix(G)
    C = cholesky(L+q*I)
    q*(C\Y)
end


function smooth_rf(G :: SimpleGraph{T},q,y :: Vector;nrep=10,variant=1) where T
    xhat = zeros(Float64,length(y));
    nr = 0;
    for indr in Base.OneTo(nrep)
        rf = random_forest(G,q)
        nr += rf.nroots
        if variant==1
            xhat += y[rf.root]
        elseif variant==2
            xhat += avg_rf(rf.root,y)
        end
    end
    (est=xhat ./ nrep,nroots=nr/nrep)
end

function smooth_rf(G :: SimpleGraph{T},q,Y :: Matrix;nrep=10,variant=1) where T
    xhat = zeros(size(Y));
    nr = 0;
    for indr in Base.OneTo(nrep)
        rf = random_forest(G,q)
        nr += rf.nroots
        if variant==1
            xhat += Y[rf.root,:]
        elseif variant==2
            xhat += avg_rf(rf.root,Y)
        end
    end
    (est=xhat ./ nrep,nroots=nr/nrep)
end


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

