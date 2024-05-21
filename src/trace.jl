"""
    partition_boundary_track(g::AbstractGraph,rf::KirchoffForest)

Helper function for computing control variates in trace estimation. Given a forest `rf` in graph `g`, this method returns a ``n\\times 1`` vector `boundary_track` where `boundary_track[i]` is the number of neighbors of node ``i`` that do not have the same root with ``i`` in `rf`.
"""
function partition_boundary_track(g::AbstractGraph,rf::KirchoffForest)
    boundary_track = zeros(nv(g))
    for i = 1 : nv(g)
        neighs = neighbors(g,i)
        boundary_track[i] = sum(rf.root[neighs] .!= rf.root[i])
    end
    boundary_track
end

"""
    root_boundary_track(g::AbstractGraph,rf::KirchoffForest)

Helper function for computing control variates in trace estimation. Given a forest `rf` in graph `g`, this method returns a vector `boundary_track` (in size of the roots of `rf`) where `boundary_track[i]` is the number of neighbors of ``i``-th root that are not rooted in ``i``-th root in `rf`.
"""
function root_boundary_track(g::AbstractGraph,rf::KirchoffForest)
    boundary_track = zeros(length(rf.roots))
    for (idx,i) in enumerate(rf.roots)
        neighs = neighbors(g,i)
        boundary_track[idx] = sum(rf.root[neighs] .!= i)
    end
    boundary_track
end

"""
    trace_estimator(g::AbstractGraph,q::Real;variant=1,α=2*q/(2*q+maximum(degree(g))),NREP=100)

Function to implement the algorithms to estimate ``tr(\\mathsf{K})``. By modifying the parameter ```variant```, one can reach [different algorithms](./trace.md). 

# Arguments
- ```g```: Input graph
- ```q```: Forest parameter ``q`` (Only implemented for scalar ``q`` for now). 

# Optional parameters 
- ```variant```: Changes the used forest estimator between four variants. 
    - ```variant=1```: Returns the estimate from number of roots (default). 
    - ```variant=2```: Returns the estimate by ``\\tilde{s}``.
    - ```variant=3```: Returns the estimate by ``\\bar{s}``.
    - ```variant=4```: Returns the estimate by ``{s}_{st}``.   
- ```α```: The update parameter for ``\\tilde{s}`` and ``\\bar{s}``. 
- ```NREP```: Number of Monte Carlo trials.

"""
function trace_estimator(g::AbstractGraph,q::Real;variant=1,α=2*q/(2*q+maximum(degree(g))),NREP=100)
    n = nv(g)
 
    if (variant == 1)
        tr_rsf = 0
        for i = 1 : NREP 
            rf = random_forest(g,q)
            tr_rsf += rf.nroots
        end
        tr_rsf /= NREP
        return tr_rsf
    elseif(variant == 2)
        tr_rsf_cv = 0
        for i = 1 : NREP 
            rf = random_forest(g,q)
            cont_var = sum(root_boundary_track(g,rf))
            tr_rsf_cv += ( rf.nroots- (α/q)*( cont_var) + α*(n - rf.nroots)  )
        end
        tr_rsf_cv /= NREP
        return tr_rsf_cv
    elseif(variant == 3)
        tr_rsf_cv = 0
        for i = 1 : NREP 
            rf = random_forest(g,q)
            part=Int.(denserank(rf.root))
            sizep = counts(part)
            diag_S = 1 ./ sizep[part]
            boundary_track = partition_boundary_track(g,rf)
            tr_rsf_cv += ( rf.nroots- (α/q)*( diag_S'*boundary_track ) + α*(n - rf.nroots)  )
        end
        tr_rsf_cv /= NREP
        return tr_rsf_cv
    elseif(variant == 4)
        d = degree(g)
        prob_vec = q ./ (q .+ d) ## Edge probabilities
        ## Gaussian approximation for Poisson-Binomial Distribution of First-Visit Roots
        μ = sum(prob_vec)
        σ = sum(prob_vec .* (1 .- prob_vec))^(1/2)
        b = Normal(μ,σ)
        ## Strat bins are chosen according to quantiles of the Gaussian approximation 
        strat_f =  quantile.(b,collect(0.2:0.2:0.8))
        n_strats = length(strat_f)+1
        strats_p = repeat([0.2],n_strats)

        strat_bins = separator2bins(strat_f,n) ## Strat bins are calculated
        # Stratification
        size_st = round.(NREP*strats_p)

        tr_per_st = zeros(n_strats)
        @inbounds for binidx = 1 : n_strats
            @inbounds for i = 1 : size_st[binidx]
                fl = cond_bernouilli_sampler(prob_vec,strat_bins[binidx,:])
                rf = random_forest_cond_first_layer(g,q,BitSet(fl);plot_flag=false)
                tr_per_st[binidx] += rf.nroots
            end
        end
        size_st[size_st .== 0] .= eps() # To avoid division by zero
        tr_per_st = (tr_per_st ./ size_st)'
        tr_st = tr_per_st * strats_p
        return tr_st
    end
end

function my_quantile(pmf::Array,p::Array)
    c = pmf[1]
    p_counter = 1
    strat_f = []
    strat_p = []
    i = 0;
    while (p_counter != length(p)+1)
        if(c > p[p_counter])
            push!(strat_f,i)
            p_counter+=1
            if(isempty(strat_p))
                push!(strat_p,c)
            else
                push!(strat_p,c - sum(strat_p))
            end
        end
        i += 1
        c += pmf[i+1]
    end
    push!(strat_p,1-sum(strat_p))
    strat_f,strat_p
end

function random_forest_cond_first_layer(G::AbstractGraph,q::Real,roots::BitSet;plot_flag=false)

    n = nv(G)
    in_tree = falses(n)
    in_tree[collect(roots)] .= true
    root = zeros(Int64,n)
    root[collect(roots)] .= collect(roots)
    nroots = length(roots)
    d = degree(G)
    skip_node = trues(n)
    next = zeros(Int64,n)
    @inbounds for i in 1:n
        u = Int64(i)
        while (!in_tree[u] )
            if(skip_node[u])
                skip_node[u] = false
                nbrs = neighbors(G, u)
                next[u] = rand(nbrs)
                u = next[u]
            else
                if ( (q+d[u])*rand() < q  )
                    in_tree[u] = true
                    push!(roots,u)
                    nroots += 1
                    root[u] = u
                    next[u] = 0
                else
                    nbrs = neighbors(G, u)
                    next[u] = rand(nbrs)
                    u = next[u]
                end
            end
        end
        r = root[u]
        #Retrace steps, erasing loops
        u = i
        while (!in_tree[u])
            root[u] = r
            in_tree[u] = true
            u = next[u]
        end
    end
    KirchoffForest(next,roots,nroots,root)
end


function separator2bins(strat_f,n)
    strat_bins = zeros(length(strat_f)+1,2)
    strat_bins[1,1] = -1
    strat_bins[1,2] = strat_f[1]
    for i = 2 : length(strat_f)
        strat_bins[i,1] = strat_f[i-1]
        strat_bins[i,2] = strat_f[i]
    end
    strat_bins[end,1] = strat_f[end]+1
    strat_bins[end,2] = n
    strat_bins
end


function cond_bernouilli_sampler(prob_vec::Array,Srange::Array)
    S = -1
    l = length(prob_vec)
    ber_vec = similar(prob_vec)
    while !(S > Srange[1] && S <= Srange[2])
        ber_vec = (rand(l) .< prob_vec)
        S = sum(ber_vec)
    end
    return findall(ber_vec .== 1)
end
