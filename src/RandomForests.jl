module RandomForests
using LightGraphs,LinearAlgebra,SparseArrays
import StatsBase.denserank,Statistics.mean
export random_forest,smooth,smooth_rf,smooth_rf_adapt

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

function smooth_rf_pcg(G :: SimpleGraph{T},q,y :: Vector;nrep=10,alpha=.5,step="fixed",nsteps=10) where T
    nr = 0;
    rf = [random_forest(G,q) for _ in 1:nrep]
    xt = y
    L = laplacian_matrix(G)
    cfun = (x) -> (q/2)*sum((x .- y).^2) + .5*x'*L*x
    gamma =alpha
    for indr in 1:nsteps
        gr = q*xt + L*xt - q*y
        dir = mean([avg_rf(r.root,gr) for r in rf])
        #dir = (q*U)
        if (step == "optimal")
            u = A*dir
            gamma = ( sum(xhat .* (A*u)) - dot(y,u)   )/dot(u,u)
            @show gamma
        elseif (step=="backtrack")
            curr = cfun(xt)
            gamma = alpha
#            while (norm(A*(xhat-gamma*dir) - y) > curr)
            while (cfun(xt-gamma*dir) > curr)
                gamma = gamma/2
            end
            @show gamma
        end
        xt -= gamma*dir
        @show cfun(xt)
    end
    (est=xt,cost = cfun(xt))
end


function smooth_rf_adapt(G :: SimpleGraph{T},q,y :: Vector;nrep=10,alpha=.5,step="fixed") where T
    nr = 0;
    rf = random_forest(G,q)
    xhat = avg_rf(rf.root,y)
#    @show xhat
    L = laplacian_matrix(G)
    cfun = (x) -> (q/2)*sum((x .- y).^2) + .5*x'*L*x
    A = (L+q*I)/q
    res = A*xhat - y
    gamma =alpha
#    @show res

    for indr in 2:nrep
        rf = random_forest(G,q)
        dir = avg_rf(rf.root,res)
        if (step == "optimal")
            u = A*dir
            gamma = ( sum(xhat .* (A*u)) - dot(y,u)   )/dot(u,u)
            @show gamma
        elseif (step=="backtrack")
            curr = cfun(xhat)
            gamma = alpha
#            while (norm(A*(xhat-gamma*dir) - y) > curr)
            while (cfun(xhat-gamma*dir) > curr)
                gamma = gamma/2
            end
            @show gamma
        end

        xhat -= gamma*dir
        res = A*xhat - y
        @show cfun(xhat)
        nr += rf.nroots
    end
    (est=xhat,nroots=nr/nrep)
end

function smooth(G :: SimpleGraph{T},q :: Float64,y :: Vector) where T
    L=laplacian_matrix(G)
    q*((L+q*I)\y)
end

function smooth(G :: SimpleGraph{T},q :: Vector,y :: Vector) where T
    L=laplacian_matrix(G)
    Q = diagm(0=>q)
    (L+Q)\(Q*y)
end

function smooth(G :: SimpleGraph{T},q :: Float64,Y :: Matrix) where T
    L=laplacian_matrix(G)
    q*((L+q*I)\Y)
end

function smooth(G :: SimpleGraph{T},q :: Vector,Y :: Matrix) where T
    L=laplacian_matrix(G)
    Q = diagm(0=>q)
    (L+Q)\(Q*Y)
end

function smooth(G :: SimpleGraph{T},q :: Float64,Y :: SparseMatrixCSC) where T
    L=laplacian_matrix(G)
    C = cholesky(L+q*I)
    q*(C\Y)
end

function smooth(G :: SimpleGraph{T},q :: Vector,Y :: SparseMatrixCSC) where T
    L=laplacian_matrix(G)
    Q = diagm(0=>q)
    C = cholesky(L+Q)
    C\(Q*Y)
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


function smooth_rf(G :: AbstractGraph,q,Y;nrep=10,variant=1,mean_correction=false) 
    xhat = zeros(size(Y));
    nr = 0;
    Ym = 0;
    if (mean_correction)
    end
    for indr in Base.OneTo(nrep)
        rf = random_forest(G,q)
        nr += rf.nroots
        if variant==1
            xhat += Y[rf.root,:]
        elseif variant==2
            xhat += avg_rf(rf.root,Y)
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

