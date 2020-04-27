#Sample a UST using a MC

struct Forest
    f :: SimpleGraph
    root :: Vector{Int}
    roots :: Set{Int}
end


function is_valid(f :: Forest)
    cc = connected_components(f.f)
    rr = unique(f.root) |> sort
    (length(cc) == length(f.roots)) && rr == sort(collect(f.roots))
end

function Forest(ff :: RandomForest)
    Forest(SimpleGraph(SimpleDiGraph(ff)),ff.root,ff.roots)
end
import LightGraphs.neighbors
neighbors(f:: Forest,u) = neighbors(f.f,u)
is_root(f:: Forest,u) = u ∈ f.roots
root(f :: Forest,u) = f.root[u]

function find_path(f :: Forest,u,w)
    path = [u]
    visited = BitSet()
    push!(visited,u)
    while length(path) > 0
        v = path[end]
        backtrack = true
        push!(visited,v)
        for i in neighbors(f,v)
            if (i == w)
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

#reassign r to be the root in a tree
function reassign_root!(f :: Forest,r)
    oldroot = f.root[r]
    pop!(f.roots,oldroot)
    push!(f.roots,r)
    f.root[f.root .== oldroot] .= r
    return 
end


function random_edge(g :: AbstractGraph)
    u=sample(vertices(g),Weights(degree(g)))
    (u,rand(neighbors(g,u)))
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

#Update according to the down/up MC
function downup!(f :: AbstractGraph,g :: AbstractGraph)
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

function find_connected(f :: Forest,u)
    path = [u]
    visited = BitSet()
    push!(visited,u)
    while length(path) > 0
        v = path[end]
        backtrack = true
        push!(visited,v)
        for i in neighbors(f,v)
            if !(i ∈ visited)
                push!(path,i)
                backtrack = false
                break
            end
        end
        if (backtrack)
            pop!(path)
        end
    end
    visited
end


#update the forest following a split (edge has disappeared)
function update_split!(f :: Forest,r1,r2)
    for r in [r1,r2]
        c = find_connected(f,r)
        for i in c
            f.root[i] = r
        end
        push!(f.roots,r)
    end
    @assert is_valid(f)
    return
end

#Update according to the up/down MC
#t needs to be a tree
function forest_downup!(f :: Forest,g :: AbstractGraph, q)
    if (rand() < nv(g)/(nv(g)+ne(g)))
        #Insert a link to Γ
        prop = rand(1:nv(g))
        if (is_root(f,prop)) #nothing happens
            return
        else
            path = find_path(f,root(f,prop),prop)
            #Now either cut one of the links in the path, or keep just one of the two roots
            l = length(path)-1
            w = 1/(q)
            if (rand() < (2*w)/(2*w+l)) #keep just one of the roots
                if (rand() < 1/2)
                    reassign_root!(f,prop)
                end
            else
                println("Splitting!")
                ii = rand(1:(l))
                rem_edge!(f.f,path[ii],path[ii+1])
                update_split!(f,root(f,prop),prop)
            end
        end
    else
        #add a random edge from g to t
        ee = random_edge(g)
        if !(ee ∈ edges(f.f))
            if (root(f,ee[1])==root(f,ee[2]))
                add_edge!(f.f,ee)
                cycle = find_cycle(f.f,ee[1])
                #remove at random from the cycle
                ii = rand(1:(length(cycle)-1))
                rem_edge!(f.f,cycle[ii],cycle[ii+1])
            else #we have a new edge between two trees
                add_edge!(f.f,ee)
                path = find_path(f,root(f,ee[1]),root(f,ee[2]))
                #Now either cut one of the links in the path, or keep just one of the two roots
                l = length(path)-1
                rts = broadcast((x)->root(f,x),ee)
                w = 1/q
                if (rand() < (2*w)/(2*w+l)) #keep just one of the roots
                    println("fusion!")
                    update_fusion!(f,rts[1],rts[2])
                else
                    ii = rand(1:(length(path)-1))
                    rem_edge!(f.f,path[ii],path[ii+1])
                    update_reassign!(f,rts[1],rts[2])
                end
            end
        end
        return
    end
end

function update_reassign!(f :: Forest,r1,r2)
    for r in [r1,r2]
        c = find_connected(f,r)
        for i in c 
            f.root[i] = r
        end
    end
    @assert is_valid(f)
end


function update_fusion!(f :: Forest,newroot,oldroot)
    c = find_connected(f,newroot)
    for i in c
        f.root[i] = newroot
    end
    pop!(f.roots,oldroot)
    @assert is_valid(f)
end

function random_tree_mc(g :: AbstractGraph,nsteps=Int(ceil(ne(g)*log(ne(g))^2)))
    t=SimpleGraph(prim_mst(g))
    for ind in 1:nsteps
        downup!(t,g)
    end
    t
end
