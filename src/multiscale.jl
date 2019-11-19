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


function smooth_ms(g :: SimpleGraph,p :: Partition,q,y)
    gred = reduced_graph(g,p)
    qvec = q*p.sizep
    y_lowfreq = propagate(p,smooth(gred,qvec,average(p,y)))
    y_lowfreq
end


