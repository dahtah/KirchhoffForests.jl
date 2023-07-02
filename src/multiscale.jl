export Hierarchy,graph,depth,HVector,smooth_subspace
function reduced_graph(g :: SimpleGraph,p :: Partition)
    gp = SimpleWeightedGraph(p.nparts)
    W = Dict{Tuple{Int,Int},Int}()
    for i in vertices(g)
        for j in neighbors(g,i)
            if i < j && (p.part[i] != p.part[j]) #avoid double counting
                pr = (p.part[i],p.part[j])
                if haskey(W,pr)
                    W[pr] += 1
                else
                    W[pr] = 1
                end
            end
        end
    end
    for (key,value) in W
        add_edge!(gp,key[1],key[2],value)
    end
    gp
end

function reduced_graph(g :: SimpleWeightedGraph,p :: Partition)
    gp = SimpleWeightedGraph(p.nparts)
    W = Dict{Tuple{Int,Int},Int}()
    for i in vertices(g)
        for j in neighbors(g,i)
            if i < j && (p.part[i] != p.part[j]) #avoid double counting
                pr = (p.part[i],p.part[j])
                if haskey(W,pr)
                    W[pr] += weights(g)[i,j]
                else
                    W[pr] = weights(g)[i,j]
                end
            end
        end
    end
    for (key,value) in W
        add_edge!(gp,key[1],key[2],value)
    end
    gp
end


function cfun(y,x,g,q)
    (q/2)*sum((y-x).^2) + .5*x'*laplacian_matrix(g)*x
end

function smooth_ms_min(g :: SimpleGraph,p :: Partition,q,y)
    Z = zeros(nv(p),p.nparts)
    for i in 1:p.nparts
        for j in 1:nv(p)
            if (p.part[j] == i)
                Z[j,i]=1
            end
        end
    end
    L = laplacian_matrix(g)
    z=q*((q*Z'*Z+Z'*L*Z)\(Z'*y))
    propagate(p,z)
    # gred = reduced_graph(g,p)
    # y_lowfreq = propagate(p,smooth(gred,q,average(p,y)))
    # y_lowfreq
end




function multigrid_solve(h ,y,q;smoother="gd",α=.1,nsteps=10)
    if (smoother=="gd")
        sfun = (g,y,x,q)-> grad_descent(g,y,x,q;α=α)
    elseif (smoother=="rf")
        sfun = (g,y,x,q)-> rf_smoother(g,y,x,q)
    end
    x = y
    cycle = (x) -> multigrid_step(h,x,y,q,0,sfun)
    for i in 1:nsteps
        x = cycle(x)
    end
    x
end

function rf_smoother(g,y,x,q)
    pp = Partition(random_forest(g,q))
    x=pp*y
    res= Vector(x .- y + laplacian_matrix(g)*x ./ q)
    (x=x,res=res)
end


# function multigrid_solve(mg ,x,y,q;α=.1,nsteps=10)
#     cycle = (x) -> multigrid_step(mg,x,y,q;α=α)
#     for i in 1:nsteps
#         x = cycle(x)
#     end
#     x
# end

struct Hierarchy
    gs :: Vector{SimpleWeightedGraph}
    ps :: Vector{Partition}
    ws :: Vector{Vector{Int}}
end



function Hierarchy(gs,ps)
    x = ones(nv(gs[1]))
    ws = [x]
    for d in 1:length(ps)
        x = sum(ps[d],x)
        push!(ws,x)
    end
    Hierarchy(gs,ps,ws)
end

function total_weights(h :: Hierarchy,d)
    @assert d <= depth(h)
    h.ws[d+1]
end
function depth(h :: Hierarchy)
    length(h.ps)
end

#graphs are numbered from 0 to depth
function graph(h::Hierarchy,i::Int)
    h.gs[i+1]
end

function partition(h::Hierarchy,i::Int)
    h.ps[i]
end

function total_weights(h::Hierarchy)
end

function Hierarchy(g :: AbstractGraph,q,depth::Int)
    gs = [SimpleWeightedGraph(g)]
    ps = Vector{Partition}()
    for i in 1:depth
        p = Partition(random_forest(g,q))
        @show p
        g = reduced_graph(g,p)
        push!(ps,p)
        push!(gs,g)
    end
    Hierarchy(gs,ps)
end

function nv(h::Hierarchy,d)
    nv(graph(h,d))
end

function show(io::IO, h::Hierarchy)
    println(io, "Multigrid hierarchy with depth $(depth(h)).")
    println(io,"Size: $([nv(h,i) for i in 0:depth(h)]) ")
end

mutable struct HVector
    v :: Vector{Float64}
    d :: Int
    h :: Hierarchy
end

function show(io::IO, hv::HVector)
    println(io, "Multigrid vector at depth $(hv.d).")
    println(io,"Value: ")
    show(io,hv.v)
end


function HVector(v,h)
    @assert length(v) == nv(h,1)
    HVector(v,0,h)
end

Base.:vec(v::HVector) = v.v

function coarsen!(v::HVector)
    @assert v.d < depth(v.h)
    v.v = sum(partition(v.h,v.d+1),vec(v))
    v.d += 1
    return 
end

function interpolate!(v::HVector)
    @assert v.d >= 1
    v.v = propagate(partition(v.h,v.d),vec(v))
    v.d -= 1
    return 
end



function multigrid_step(h,x0,y,q,d,smoother,size_exact=40)
    x = x0
    g = graph(h,d)
    if (length(x) < size_exact)
        x = smooth(g,q,y)
    else
        x,res = smoother(g,y,x,q)
        if (d<depth(h))
            r = HVector(res,d,h)
            coarsen!(r)
            qp = q*total_weights(h,d+1)
            out =
                multigrid_step(h,zeros(length(qp)),r.v,q,d+1,smoother,size_exact)
            r.v = out./total_weights(h,d+1)
            interpolate!(r)
            x = x-vec(r)
        end
    end
    x
end


# function multigrid_step(h,x0,y,q,d;α=.1,size_exact=40)
#     x = x0
#     g = graph(h,d)
#     if (length(x) < size_exact)
#         x = smooth(g,q,y)
#     else
#         x,res = grad_descent(g,y,x,q;α=α)
#         if (d<depth(h))
#             r = HVector(res,d,h)
#             coarsen!(r)
#             qp = q*total_weights(h,d+1)
#             out =
#                 multigrid_step(h,zeros(length(qp)),r.v,q,d+1;α=α,size_exact=size_exact)
#             r.v = out./total_weights(h,d+1)
#             interpolate!(r)
#             x = x-vec(r)
#         end
#     end
#     x
# end

function smooth_subspace(g,V,q,y)
    M = Matrix(q*V'*V + V'*laplacian_matrix(g)*V)
    q*(M\(V'*y))
end


function grid_partition(n)
    Partition(repeat(1:(n ÷ 2),inner=2),n ÷ 2,2*ones(Int,n ÷ 2))
end

function mgrid_hierarchy_1d(n,depth)

    ns = n .÷ (2 .^(0:(depth)))
    @assert all(rem.(n,(2 .^(0:(depth)))) .== 0)
    gs = [SimpleWeightedGraph(grid([i])) for i in ns]
    ps= [grid_partition(i) for i in ns[1:(end-1)] ]
    Hierarchy(gs,ps)
end

function smooth_ms(g :: SimpleGraph,p :: Partition,q,y)
    gred = reduced_graph(g,p)
    qvec = q*p.sizep
    y_lowfreq = propagate(p,smooth(gred,qvec,average(p,y)))
    y_lowfreq
end


