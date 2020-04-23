#Sample a UST using a MC

function random_edge(g :: AbstractGraph)
    u=sample(vertices(g),Weights(degree(g)))
    (u,rand(neighbors(g,u)))
end

function fc(t :: AbstractGraph,u)
    t2 = copy(t)
    @assert t == t2
    stack = neighbors(t,u)
    path = [u]
    while length(stack) > 0
        @show stack
        nxt = pop!(stack)
        @show nxt
        @assert t == t2
        push!(path,nxt)
        @show nxt,path
        @assert t == t2
        if (nxt==u)
            return path
        else
            if (degree(t,nxt) == 1) #leaf
                pop!(path)
            else
                dd = setdiff(neighbors(t,nxt),path)
                for d in dd
                    push!(stack,d)
                end
            end
        end
    end
    path
end


function find_cycle(t :: AbstractGraph,u)
#    stack = copy(neighbors(t,u))
    path = [u]
    visited = BitSet()
    push!(visited,u)
    while length(path) > 0
        v = path[end]
        backtrack = true
        push!(visited,v)
        for i in neighbors(t,v)
            if (length(path) > 2 && i == u)
                push!(path,i)
                return path
            elseif !(i ∈ visited)
                push!(path,i)
                backtrack = false
                break
            end
        end
        if (backtrack)
            pop!(path)
        end
    end
    path
end

#Update according to the up/down MC
#t needs to be a tree
function updown!(t :: AbstractGraph,g :: AbstractGraph)
    #add a random edge from g to t
    ee = random_edge(g)
    if !(ee ∈ edges(t))
        add_edge!(t,ee)
        cycle = find_cycle(t,ee[1])
        #remove at random from the cycle
        ii = rand(1:(length(cycle)-1))
        rem_edge!(t,cycle[ii],cycle[ii+1])
    end
    return
end

function random_tree_mc(g :: AbstractGraph,nsteps=Int(ceil(ne(g)*log(ne(g))^2)))
    t=SimpleGraph(prim_mst(g))
    for ind in 1:nsteps
        updown!(t,g)
    end
    t
end
