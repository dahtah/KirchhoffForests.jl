#all j's are relabeled i
function fuse!(p :: Partition,i,j)
    for ind in 1:length(p.part)
        if p.part[ind] == j
            p.part[ind] = i
        end
    end
    p.sizep[i] +=     p.sizep[j] 
    p.sizep[j] = 0
end

function rexp(rate)
    -log(rand())/rate
end

function wsample(pr)
    s = 0
    u = rand()
    i = 1
    cr = sum(pr)
    while i <= length(pr)
        s+=pr[i]
        if (s/cr >= u)
            break
        end
        i+=1
    end
    i
end

function rclock(rates)
    cr = sum(rates)
    tt = -log(rand())/cr
    s = 0
    u = rand()
    i = 1
    while i <= length(rates)
        s+=rates[i]
        if (s/cr >= u)
            break
        end
        i+=1
    end
    (time=tt,which=i)
end

function agproc(g :: SimpleGraph,q)
    n = nv(g)
    p = Partition(1:n,n,ones(n))
    free_roots = BitSet(1:n)
    frozen_roots = BitSet()
    d = degree(g)

    while length(free_roots) > 0
        @debug free_roots,frozen_roots
        rr = collect(free_roots)
        nfree = length(free_roots)
        sdr = sum(d[rr])
        pr_freeze = (nfree*q)/(nfree*q + sdr)
        if rand() < pr_freeze
            fr = rand(rr)
            @debug "Freezing $(fr)"
            push!(frozen_roots,fr)
            delete!(free_roots,fr)
        else #move event
            mv = rr[wsample(d[rr])]
            @debug "Moving $(mv)"
            nxt = random_successor(g,mv)
            @assert p.part[mv] == mv
            if (p.part[nxt] != mv) #gets eaten
                @debug " $(mv) got eaten by $(p.part[nxt])"
                #fuse!(p,p.part[nxt],p.part[mv])
                fuse!(p,p.part[nxt],p.part[mv])
                delete!(free_roots,mv)
            end
        end
    end
    frozen_roots,p
end

function bvec(v :: BitSet,n)
    ind = zeros(Int,n)
    for s in v
        ind[s] = 1
    end
    ind
end

function check_prob(g,q,nrep=100)
    n = nv(g)
    s = zeros(n)
    for ind in 1:nrep
        r,p = RandomForests.agproc(g,q)
        s += bvec(r,n)
    end
    s/nrep
end
