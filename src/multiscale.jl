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

function multigrid_solve(mg ,x,y,q;α=.1,nsteps=10)
    cycle = (x) -> multigrid_step(mg,x,y,q;α=α)
    for i in 1:nsteps
        x = cycle(x)
    end
    x
end

function multigrid_step(mg ,x0,y,q;α=.1,size_exact=5)
#    A = (1/q)*(q*I+laplacian_matrix(L))
    g = mg[1][1]
    p = mg[1][2]
    
    L = laplacian_matrix(g)
    gr = Vector(L*x0 .+ q.*(x0 .- y))
    @assert length(gr)==nv(g)
    println("Blah blah")
    if (nv(g) <= size_exact)
        println("Exact solve")
        return vec(smooth(g,q,y))
    else
        println( "Smoothing")
        #Apply smoother
        x = Vector(x0 .- α*(gr)./diag(L))
        
        #    @show L*x,q.*(x-y)
        println("Computing gradient")
        gr2 = Vector(L*x .+ q.*(x-y))
#        @show size(gr2),nv(g)
        if (length(mg)>1)
            # @show average(p,gr2) 
            smooth_res = vec(average(p,gr2))
            qnew = vec(average(p,q)).*p.sizep
            # @show qnew
            corrected_res = multigrid_step(mg[2:end],zeros(p.nparts),smooth_res,qnew;α=α)
           # @show corrected_res
            @assert size(x)==size(propagate(p,corrected_res))
            #@show x,propagate(p,corrected_res)
            x-= propagate(p,corrected_res)
        end
        return x
    end
end

function grid_partition(n)
    Partition(repeat(1:(n ÷ 2),inner=2),n ÷ 2,2*ones(Int,n ÷ 2))
end

function mgrid_hierarchy_1d(n,depth)
    ns = n .÷ (2 .^(0:(depth - 1)))
     [(grid([i]),grid_partition(i)) for i in ns ]
end

function smooth_ms(g :: SimpleGraph,p :: Partition,q,y)
    gred = reduced_graph(g,p)
    qvec = q*p.sizep
    y_lowfreq = propagate(p,smooth(gred,qvec,average(p,y)))
    y_lowfreq
end


