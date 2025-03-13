#Coupled forests are a variant of Kirchhoff forests introduced in
#Avena & Gaudillière (2017). The algorithm generates a sequence of coupled KFs for a decreasing sequence q_1 >= q_2 >= ... >= q_m. At each time point the coupled KF is distributed according to a KF with parameter q_t



# This file provides 5 different implementations of the coupled forests:
# - "Standard" uses the structure RfState and computes the coupled forests in the simplest way without
# keeping updated the vector root at all times. For each q in the grid, it stalls the coupled forest algorithm to 
# compute the root vector and then the number of roots of roots. In fact, these root computations take more than half the 
# time on classical examples
# - "Standard_PQ" is the same but with a priority queue to keep in memory all qstars that "killed" all roots. This 
# makes it easier to know which node to awaken: the one associated to the largest qstar. On the counterpart, it necessitates to 
# keep in memory and in a sorted order a rather large list of qs at all times.
# - "Leaves" is a more intricate implementation that tries to keep updated the vector root at all times. In order 
# to do this, the information of a merge for instance needs to go back up the tree which is not easy to do. We 
# decide in this implem to keep the information of all trees by storing the vector Next as usual but also all the 
# leaves of each tree
# - "Leaves_PQ" is the same but with priority queue
# - "Independent_Wilson" is a "fake" coupled forest algorithm: it just runs independent Wilsons at each q

# on many examples, "Standard" should be the one to use. Unless dmax is an outlier compared to the degree distribution in 
# this case use "Standard_PQ"; or unless qmin is very small or length(qrange) is very large, in which case use "Leaves" 
# or "Leaves_PQ" 

#TODO:
#Add an "observer" mode where you can provide an observer that computes some forest quantity at fixed q
#Make all the types parametric so we can use sthg else than hard-coded Int64s and Floats. 

using Distributions

mutable struct RfState
    deg :: Array{Float64,1} #degrees
    dmax :: Float64 #max degree
    short_awakenings :: Int64 #number of "short awakenings"
    root :: Vector{Int64} #root[i] is the root of node i
    is_root :: Array{Bool,1} #is_root
    root_set :: Set{Int64} #set of roots
    root_counter :: Int64 #root_counter
    root_is_correct::Bool # a Boolean to flag when state.root is correct
    is_alive :: Array{Bool,1} #is alive
    alive_nodes :: Vector{Int64} #vector of alive nodes
    next :: Array{Int64,1} #vector next encoding the RF
end

mutable struct RfState_PQ
    deg :: Array{Float64,1} #degrees
    dmax :: Float64 #max degree
    list_q :: PriorityQueue{Int64, Float64, Base.Order.ReverseOrdering{Base.Order.ForwardOrdering}} #priority queue to store the values of q
    root :: Vector{Int64} #root[i] is the root of node i
    is_root :: Array{Bool,1} #is_root
    root_set :: Set{Int64} #set of roots
    root_counter :: Int64 #root_counter
    root_is_correct::Bool # a Boolean to flag when state.root is correct
    is_alive :: Array{Bool,1} #is alive
    alive_nodes :: Vector{Int64} #vector of alive nodes
    next :: Array{Int64,1} #vector next encoding the RF
end

mutable struct RfState_leaves
    deg :: Array{Float64,1}
    dmax :: Float64
    false_awakenings :: Int64
    root :: Array{Int64,1}
    root_counter :: Int64
    next :: Array{Int64,1}
    nextsave :: Array{Int64,1}
    leaf_sets :: Dict{Int64, Array{Int64,1}}
    is_leaf :: Array{Bool,1}
    next_is_already_sampled :: Array{Bool,1}
end

mutable struct RfState_leaves_PQ
    deg :: Array{Float64,1}
    dmax :: Float64
    list_q :: PriorityQueue{Int64, Float64, Base.Order.ReverseOrdering{Base.Order.ForwardOrdering}} #priority queue to store the values of q
    root :: Array{Int64,1}
    root_counter :: Int64
    next :: Array{Int64,1}
    nextsave :: Array{Int64,1}
    leaf_sets :: Dict{Int64, Array{Int64,1}}
    is_leaf :: Array{Bool,1}
    next_is_already_sampled :: Array{Bool,1}
end

function RfState(g::SimpleGraph{Int64})
    n = nv(g)
    dmax = convert(Float64, maximum(degree(g)))
    return RfState(
            convert.(Float64, degree(g)), #deg
            dmax, #max degree
            0, #number of rejections: number of "short awakenings"
            Vector{Int64}(undef, n), # vector root
            falses(n), #is_root
            Set{Int64}(), #set of roots
            0, #root_counter
            false, #root_is_correct
            falses(n), #is alive
            Vector{Int64}(), #vector of alive nodes
            zeros(Int64, n) #next
            )
end

function RfState_PQ(g::SimpleGraph{Int64})
    n = nv(g)
    dmax = convert(Float64, maximum(degree(g)))
    return RfState_PQ(
            convert.(Float64, degree(g)), #deg
            dmax, #max degree
            PriorityQueue{Int64, Float64}(Base.Order.Reverse), #list_q
            Vector{Int64}(undef, n), # vector root
            falses(n), #is_root
            Set{Int64}(), #set of roots
            0, #root_counter
            false, #root_is_correct
            falses(n), #is alive
            Vector{Int64}(), #vector of alive nodes
            zeros(Int64, n) #next
            )
end

function RfState_leaves(g::SimpleGraph{Int64})
    n = nv(g)
    dmax = convert(Float64, maximum(degree(g)))
    return RfState_leaves(
            convert.(Float64, degree(g)), #deg
            dmax, #max degree
            0, #number of rejections: number of "short awakenings"
            zeros(Int64, n+1), #root
            0, #root_counter
            zeros(Int64, n), #next
            zeros(Int64, n+1), #nextsave
            Dict{Int64, Array{Int64,1}}(), # dictionnary that associates each root to its leaf set
            zeros(Bool, n), # is_leaf
            zeros(Bool, n) #next_is_already_sampled
            )
end

function RfState_leaves_PQ(g::SimpleGraph{Int64})
    n = nv(g)
    dmax = convert(Float64, maximum(degree(g)))
    return RfState_leaves_PQ(
            convert.(Float64, degree(g)), #deg
            dmax, #max degree
            PriorityQueue{Int64, Float64}(Base.Order.Reverse), #list_q
            zeros(Int64, n+1), #root
            0, #root_counter
            zeros(Int64, n), #next
            zeros(Int64, n+1), #nextsave
            Dict{Int64, Array{Int64,1}}(), # dictionnary that associates each root to its leaf set
            zeros(Bool, n), # is_leaf
            zeros(Bool, n) #next_is_already_sampled
            )
end

"""
This function collects an estimate of the first n_moments of the distribution q / q + λ, for all qs in qrange; where λ 
    is the spectrum of a graph g. This estimate is made with coupled forests.

Usage
----------
```moments = collect_moments_coupled_forests(g, qrange, n_moments, s, algo)```

Entry
----------
* ```g``` : a graph (type SimpleGraph generated with Graphs.jl)
* ```qrange``` : the range of qs at which to estimate moments
* ```n_moments``` : the number of moments to estimate
* ```s``` : the number of Monte-Carlo estimates (the number of forests drawn is thus s * n_moments, as we need k forests to estimate the k-th moment)
* ```algo''': either ":standard", ":standard_pq", ":leaves", ":leaves_pq", "Independent_Wilson". Use "Standard" by default. Use "Standard_PQ" if dmax is really an outlier of the degree distribution. Use "Leaves" or "Leaves_PQ" if qmin is very small or if length(qrange) is very large. Don't use "Independent_Wilson" 

Returns
-------
```moments``` : returns a vector of size(qrange). moments[i].y are the moments computef dor qrange[i]. moments[i].var are bounds of the variance of these estimates.
"""
function collect_moments_coupled_forests(g::AbstractGraph, qrange :: AbstractVector, n_moments, s, algo = :standard)
    #= if s > 4 # if the empirical variance has a reasonable chance of being approximated
        return collect_moments_coupled_forests_with_empirical_var(g, qrange, n_moments, s, algo)
    end #if not, we only use a known bound on the variance: =#
    n = nv(g)
    N_q = length(qrange)
    est_moments = mean((coupled_forests_moments(g, qrange, n_moments, algo) for _ in 1:s))
    est_moments[est_moments .< 1.0] .= 1.0 # this is because we know that all moments are necessarily above 1.0 !
    est_moments = est_moments ./ n
    
    var_bound = (1/s)*(1/n) .* est_moments #Compute an approximate bound on the variance (Alex's bound)
    var_bound = var_bound ./ binomial.(n_moments, 1:n_moments) # this is because we compute all combinations of k in nm

    moments = Vector{@NamedTuple{y::Vector{Float64}, var::Vector{Float64}}}(undef,N_q)
    if n_moments>1
        for i=1:N_q
            moments[i] = (y = est_moments[:,i], var = var_bound[:,i])
        end
    else
        for i=1:N_q
            moments[i] = (y = [est_moments[i]], var = [var_bound[i]])
        end
    end
    return moments
end

"""
This function is the same as collect_moments_coupled_forests. The difference is that the variance bound is partially estimated by the empirical variance of the estimators (s needs to be large enough however)

Usage
----------
```moments = collect_moments_coupled_forests_with_empirical_var(g, qrange, n_moments, s, algo)```

Entry
----------
* ```g``` : a graph (type SimpleGraph generated with Graphs.jl)
* ```qrange``` : the range of qs at which to estimate moments
* ```n_moments``` : the number of moments to estimate
* ```s``` : the number of Monte-Carlo estimates (the number of forests drawn is thus s * n_moments, as we need k forests to estimate the k-th moment)
* ```algo''': either "Standard", "Standard_PQ", "Leaves", "Leaves_PQ", "Independent_Wilson". Use "Standard" by default. Use "Standard_PQ" if dmax is really an outlier of the degree distribution. Use "Leaves" or "Leaves_PQ" if qmin is very small or if length(qrange) is very large. Don't use "Independent_Wilson" 

Returns
-------
```moments``` : returns a vector of size(qrange). moments[i].y are the moments computed for qrange[i]. moments[i].var are estimates of the variance of these moments.
"""
function collect_moments_coupled_forests_with_empirical_var(g::SimpleGraph{Int64}, qrange :: AbstractVector, n_moments :: Integer, s :: Integer, alg = :standard)
    n = nv(g)
    N_q = length(qrange)

    # Preallocate accumulators for the sum and sum of squares
    sum_moments = zeros(Float64, n_moments, N_q)
    sumsq_moments = zeros(Float64, n_moments, N_q)

    @inbounds for i in 1:s
        # Call coupled_forests for each Monte Carlo sample
        sample_moments = coupled_forests(g, qrange, n_moments, algo)
        
        # Accumulate the sample in-place
        sum_moments .+= sample_moments
        sumsq_moments .+= sample_moments .^ 2
    end

    # Compute the mean over the s samples.
    est_moments = sum_moments ./ s

    # Enforce the lower bound: every moment is at least 1.0
    est_moments[est_moments .< 1.0] .= 1.0

    # Normalize by the number of vertices
    est_moments .= est_moments ./ n

    # Compute the variance:
    # Note: This uses the population variance formula, i.e. E[X²] - (E[X])².
    # Then, we divide by s*n^2.
    var_moments = (sumsq_moments ./ s .- (sum_moments ./ s) .^ 2) ./ (s * n^2)
    #var_moments = var_moments ./ binomial.(n_moments, 1:n_moments) # this is because we compute all combinations of k in nm

    var_crude_bound = (1/s)*(1/n) .* est_moments #Compute an approximate bound on the variance (Alex's bound)
    var_crude_bound = var_crude_bound ./ binomial.(n_moments, 1:n_moments) # this is because we compute all combinations of k in nm

    indx = var_moments .== 0
    var_moments[indx] .= 1e-18 #var_crude_bound[indx]

    var_bound = min.(var_crude_bound, 2 .*var_moments) #take the minimum of the two "bounds"
    moments = Vector{@NamedTuple{y::Vector{Float64}, var::Vector{Float64}}}(undef,N_q)
    if n_moments>1
        for i=1:N_q
            moments[i] = (y = est_moments[:,i], var = var_bound[:,i])
        end
    else 
        for i=1:N_q
            moments[i] = (y = [est_moments[i]], var = [var_bound[i]])
        end
    end
    return moments
    #= begin
        plot((est_moments.*n .+ 10)', yaxis=:log, yticks=[1, 5, 10, 100, 1000, 5000])
        plot!((est_moments.*n .+ 10)' .+ 3 .*sqrt.(var_bound .* n^2)', linestyle=:dot)
        plot!((est_moments.*n .+ 10)' .- 3 .*sqrt.(var_bound .* n^2)', linestyle=:dot)
    end =#
end

"""
This function estimates the first n_moments of the distribution q / q + λ, for all qs in qrange; where λ 
    is the spectrum of a graph g. This estimate is made with coupled forests.

Usage
----------
```moments = coupled_forests(g, qrange, n_moments, algo)```

Entry
----------
* ```g``` : a graph (type SimpleGraph generated with Graphs.jl)
* ```qrange``` : the range of qs at which to estimate moments
* ```n_moments``` : the number of moments to estimate
* ```algo''': either "Standard", "Standard_PQ", "Leaves", "Leaves_PQ", "Independent_Wilson". Use "Standard" by default. Use "Standard_PQ" if dmax is really an outlier of the degree distribution. Use "Leaves" or "Leaves_PQ" if qmin is very small or if length(qrange) is very large. Don't use "Independent_Wilson" 

Returns
-------
```moments``` : returns a matrix of n_moments x size(qrange). moments[i,j] is the i-th moment of the distribution q / q + λ for q = qrange[j].
"""
function coupled_forests_moments(g::SimpleGraph, q :: AbstractVector, nmoments :: Integer,algo  = :standard)
#    @show algo
    if algo == :indep #independent Wilson at each q
        @info "Running indep. replicates"
        return fake_coupled_forests(g, q, nmoments)
    end
    nq = length(q)
    qmax = q[end]
    qmin = q[1]
    qstar = fill(qmax, nmoments)
    # initialize:
    if algo == :standard
        state = Vector{RfState}(undef,nmoments)
        @inbounds for j=1:nmoments
            state[j] = RfState(g)
        end
        id = collect(1:nv(g))
    elseif algo == :standard_pq
        state = Vector{RfState_PQ}(undef,nmoments)
        @inbounds for j=1:nmoments
            state[j] = RfState_PQ(g)
        end
        id = collect(1:nv(g))
    elseif algo == :leaves
        state = Vector{RfState_leaves}(undef,nmoments)
        @inbounds for j=1:nmoments
            state[j] = RfState_leaves(g)
        end
        id = collect(1:nv(g)+1)
    elseif algo == :leaves_pq
        state = Vector{RfState_leaves_PQ}(undef,nmoments)
        @inbounds for j=1:nmoments
            state[j] = RfState_leaves_PQ(g)
        end
        id = collect(1:nv(g)+1)
    else 
        error("Algorithm is not recognized")
    end
    
    # run a first wilson with q = qmax:
    @inbounds for j=1:nmoments
        state[j] = wilson!(g, qstar[j], state[j]);
    end

    # save results:
    if nmoments == 1
        results = Vector{Float64}(undef,nq)
        results[nq] = state[1].root_counter
    else
        results = Matrix{Float64}(undef,nmoments,nq)
        results[:,nq] = get_n_self_roots(nv(g), state, id)
    end

    # find the next q and the associated root to awaken
    r = Vector{Int64}(undef,nmoments)
    @inbounds for j=1:nmoments
        r[j], qstar[j] = find_the_root_to_awaken(qstar[j], state[j])
    end

    for q_counter = 1:nq-1 # run until we went through all desired values of q in vector q
        threshold = q[end - q_counter] 
        @inbounds for j=1:nmoments # do the following for all forests
            while qstar[j] > threshold #run the coupled forest algorithm until the current "qstar" crosses the desired q
                state[j] = reboot!(g, qstar[j], r[j], state[j]) #awaken that root
                r[j], qstar[j] = find_the_root_to_awaken(qstar[j], state[j]) # find the next q and the associated root to awaken
            end
        end
        # at this point all nmoments forests are stalled at q[end-q_counter]
        if nmoments == 1
            results[nq-q_counter] = state[1].root_counter
        else
            results[:,nq-q_counter] = get_n_self_roots(nv(g),state,id)
        end
    end
    return results
end

"""
This function estimates the first n_moments of the distribution q / q + λ, for all qs in qrange; where λ 
    is the spectrum of a graph g. This estimate is made by running independent Wilsons for each q in qrange. 
    This is NOT a coupled forest algorithm. The estimates are however valid and unbiased (it just usually takes 
    longer than a coupled forest variant, as soon as qrange is of length typically larger than 3)

Usage
----------
```moments = fake_coupled_forests(g, qrange, n_moments)```

Entry
----------
* ```g``` : a graph (type SimpleGraph generated with Graphs.jl)
* ```qrange``` : the range of qs at which to estimate moments
* ```n_moments``` : the number of moments to estimate

Returns
-------
```moments``` : returns a matrix of n_moments x size(qrange). moments[i,j] is the i-th moment of the distribution q / q + λ for q = qrange[j].
"""
function fake_coupled_forests(g::SimpleGraph{Int64}, qrange :: Vector{Float64}, nmoments :: Int64)
    # this is NOT a coupled forest algorithm: this samples independently forests at each q in qrange, via Wilson's algorithm
    nq = length(qrange)
    if nmoments == 1 # then root_counter is enough
        results = Vector{Float64}(undef, nq)
        for i =1:nq
            results[i], _ = wilson(g, qrange[i])
        end
    else # then it is much more complicated: we need the vector roots!
        results = Matrix{Float64}(undef, nmoments, nq)
        root_counter = zeros(Int64, nmoments)
        root = Vector{Vector{Int64}}(undef, nmoments)
        id = collect(1:nv(g))
        for i =1:nq
            for j=1:nmoments
                root_counter[j], root[j] = wilson(g, qrange[i])
            end
            results[:,i] = get_n_self_roots(nv(g), root_counter, root, id)
        end
    end
    return results
end

"""
This function computes the number of "self roots-of-roots", and by doing so estimates moments of q / q + λ. For n_moments it 
    takes as entry root vectors of n_moments independent forests sampled at the same q. This function is specific for the "Independent_Wilson" implem.

Usage
----------
```results = get_n_self_roots(n, root_counter, root, id)```

Entry
----------
* ```n``` : number of nodes of the graph
* ```root_counter``` : vector os size n_moments. root_counter[j] is the number of roots of forest j. 
* ```root``` : vector of root vectors. root[j] is the root vector of forest j.
* ```id''' : should be id = collect(1:n). Leaving it as an input avoids to compute 1:n every time we run the function

Returns
-------
```results``` : returns a vector of n_moments. 
"""
function get_n_self_roots(n::Int64, root_counter::Vector{Int64}, root_vectors::Vector{Vector{Int64}}, id::Vector{Int64}) 
    nm = length(root_counter)
    n_selfroots = zeros(Float64, nm)

    # Recursive function to traverse all combinations.
    # 'last' is the last index used in the combination,
    # 'v' is the composed mapping so far,
    # 'size' is the current size of the combination.
    function rec(last::Int, v::Vector{Int}, size::Int)
        # When the combination is nonempty, update the count.
        if size > 1
            # Count fixed points: those indices i for which v[i]==i, then subtract 1.
            n_selfroots[size] += count(v .== id)
        end
        # Try to extend the combination.
        for i in (last+1):nm
            # Compute the new composition.
            new_v = root_vectors[i][v]
            rec(i, new_v, size + 1)
        end
    end
    
    @inbounds for j=1:nm
        rec(j, root_vectors[j], 1)
    end
    
    # Divide each total by the number of combinations of that size.
    @inbounds for j in 1:nm
        n_selfroots[j] /= binomial(nm, j)
    end
    
    # for only singleton forests, we have already the root_counters:
    n_selfroots[1] = mean(root_counter)
    return n_selfroots
end

"""
This function computes the number of "self roots-of-roots", and by doing so estimates moments of q / q + λ. For n_moments moments it 
    takes as entry n_moments independent forests (in vector 'state') sampled at the same q. This function is specific for the "Standard" implem.

Usage
----------
```results = get_n_self_roots(n, state, id)```

Entry
----------
* ```n``` : number of nodes of the graph
* ```state``` : vector of size n_moments. state[j] is the RfState of forest j.
* ```id''' : should be id = collect(1:n). Leaving it as an input avoids to compute 1:n every time we run the function

Returns
-------
```results``` : returns a vector of n_moments. 
"""
function get_n_self_roots(n::Int64, state::Vector{RfState}, id::Vector{Int64})
    nm = length(state)

    if !state[1].root_is_correct #this only happens at qmax after the first initial run of Wilson
        @inbounds for j=1:nm
            state[j] = get_root_vector!(n, state[j])
        end
    end

    n_selfroots = zeros(Float64, nm)

    # Recursive function to traverse all combinations.
    # 'last' is the last index used in the combination,
    # 'v' is the composed mapping so far,
    # 'size' is the current size of the combination.
    function rec(last::Int, v::Vector{Int}, size::Int)
        # update the count.
        if size > 1
            # Count fixed points: those indices i for which v[i]==i.
            n_selfroots[size] += count(v .== id)
        end
        # Try to extend the combination.
        for i in (last+1):nm
            # Compute the new composition.
            new_v = state[i].root[v]
            rec(i, new_v, size + 1)
        end
    end
    
    @inbounds for j=1:nm
        rec(j, state[j].root, 1)
    end
    
    # Divide each total by the number of combinations of that size.
    @inbounds for j in 2:nm
        n_selfroots[j] /= binomial(nm, j)
    end
    
    # for only singleton forests, we have already the root_counters:
    n_selfroots[1] = mean(s.root_counter for s in state)
    
    return n_selfroots
end

"""
This function computes the root vector of a given forest sampled at q This function is specific for the "Standard" implem.

Usage
----------
```state = get_root_vector!(n, state)```

Entry
----------
* ```n``` : number of nodes of the graph
* ```state``` : vector of size n_moments. state[j] is the RfState of forest j.

Returns
-------
```state``` : the same forest with state.root that is now updated to be the current root vector
"""
function get_root_vector!(n::Int64, state::RfState)
    root_is_found = copy(state.is_root)

    for i = 1:n
        current = i
        
        # Traverse the forest (via next) to find the root
        while !root_is_found[current]
            current = state.next[current]
        end
        root_of_tree = state.root[current]

        # Propagate the root for all nodes in the path
        current = i
        while !root_is_found[current]
            state.root[current] = root_of_tree
            root_is_found[current] = true
            current = state.next[current]
        end
    end
    state.root_is_correct = true
    return state
end

"""
This function computes the number of "self roots-of-roots", and by doing so estimates moments of q / q + λ. For n_moments moments it 
    takes as entry n_moments independent forests (in vector 'state') sampled at the same q. This function is specific for the "Standard_PQ" implem.

Usage
----------
```results = get_n_self_roots(n, state, id)```

Entry
----------
* ```n``` : number of nodes of the graph
* ```state``` : vector of size n_moments. state[j] is the RfState_PQ of forest j.
* ```id''' : should be id = collect(1:n+1). Leaving it as an input avoids to compute 1:n+1 every time we run the function

Returns
-------
```results``` : returns a vector of n_moments moments.
"""
function get_n_self_roots(n::Int64, state::Vector{RfState_PQ}, id::Vector{Int64})
    nm = length(state)

    if !state[1].root_is_correct #this only happens at qmax after the first initial run of Wilson
        @inbounds for j=1:nm
            state[j] = get_root_vector!(n, state[j])
        end
    end

    n_selfroots = zeros(Float64, nm)

    # Recursive function to traverse all combinations.
    # 'last' is the last index used in the combination,
    # 'v' is the composed mapping so far,
    # 'size' is the current size of the combination.
    function rec(last::Int, v::Vector{Int}, size::Int)
        # update the count.
        if size > 1
            # Count fixed points: those indices i for which v[i]==i.
            n_selfroots[size] += count(v .== id)
        end
        # Try to extend the combination.
        for i in (last+1):nm
            # Compute the new composition.
            new_v = state[i].root[v]
            rec(i, new_v, size + 1)
        end
    end
    
    @inbounds for j=1:nm
        rec(j, state[j].root, 1)
    end
    
    # Divide each total by the number of combinations of that size.
    @inbounds for j in 2:nm
        n_selfroots[j] /= binomial(nm, j)
    end
    
    # for only singleton forests, we have already the root_counters:
    n_selfroots[1] = mean(s.root_counter for s in state)
    
    return n_selfroots
end

"""
This function computes the root vector of a given forest sampled at q This function is specific for the "Standard_PQ" implem.

Usage
----------
```state = get_root_vector!(n, state)```

Entry
----------
* ```n``` : number of nodes of the graph
* ```state``` : vector of size n_moments. state[j] is the RfState_PQ of forest j.

Returns
-------
```state``` : the same forest with state.root that is now updated to be the current root vector
"""
function get_root_vector!(n::Int64, state::RfState_PQ)
    root_is_found = copy(state.is_root)

    for i = 1:n
        current = i
        
        # Traverse the forest (via next) to find the root
        while !root_is_found[current]
            current = state.next[current]
        end
        root_of_tree = state.root[current]

        # Propagate the root for all nodes in the path
        current = i
        while !root_is_found[current]
            state.root[current] = root_of_tree
            root_is_found[current] = true
            current = state.next[current]
        end
    end
    state.root_is_correct = true
    return state
end

"""
This function computes the number of "self roots-of-roots", and by doing so estimates moments of q / q + λ. For n_moments moments it 
    takes as entry n_moments independent forests (in vector 'state') sampled at the same q. This function is specific for the "Leaves" implem.

Usage
----------
```results = get_n_self_roots(n, state, id)```

Entry
----------
* ```n``` : number of nodes of the graph
* ```state``` : vector of size n_moments. state[j] is the RfState_Leaves of forest j.
* ```id''' : should be id = collect(1:n+1). Leaving it as an input avoids to compute 1:n+1 every time we run the function

Returns
-------
```results``` : returns a vector of n_moments moments.
"""
function get_n_self_roots(n::Int, state::Vector{RfState_leaves}, id::Vector{Int64}) 
    nm = length(state)
    n_selfroots = zeros(Float64, nm)

    # Recursive function to traverse all combinations.
    # 'last' is the last index used in the combination,
    # 'v' is the composed mapping so far,
    # 'size' is the current size of the combination.
    function rec(last::Int, v::Vector{Int}, size::Int)
        # When the combination is nonempty, update the count.
        if size > 1
            # Count fixed points: those indices i for which v[i]==i, then subtract 1.
            n_selfroots[size] += count(v .== id) - 1
        end
        # Try to extend the combination.
        for i in (last+1):nm
            # Compute the new composition.
            new_v = state[i].root[v]
            rec(i, new_v, size + 1)
        end
    end
    
    @inbounds for j=1:nm
        rec(j, state[j].root, 1)
    end
    
    # Divide each total by the number of combinations of that size.
    @inbounds for j in 2:nm
        n_selfroots[j] /= binomial(nm, j)
    end
    
    # for only singleton forests, we have already the root_counters:
    n_selfroots[1] = mean(s.root_counter for s in state)
    
    return n_selfroots
end

"""
This function computes the number of "self roots-of-roots", and by doing so estimates moments of q / q + λ. For n_moments moments it 
    takes as entry n_moments independent forests (in vector 'state') sampled at the same q. This function is specific for the "Leaves_PQ" implem.

Usage
----------
```results = get_n_self_roots(n, state, id)```

Entry
----------
* ```n``` : number of nodes of the graph
* ```state``` : vector of size n_moments. state[j] is the RfState_Leaves_PQ of forest j.
* ```id''' : should be id = collect(1:n+1). Leaving it as an input avoids to compute 1:n+1 every time we run the function

Returns
-------
```results``` : returns a vector of n_moments moments.
"""
function get_n_self_roots(n::Int, state::Vector{RfState_leaves_PQ}, id::Vector{Int64}) 
    nm = length(state)
    n_selfroots = zeros(Float64, nm)

    # Recursive function to traverse all combinations.
    # 'last' is the last index used in the combination,
    # 'v' is the composed mapping so far,
    # 'size' is the current size of the combination.
    function rec(last::Int, v::Vector{Int}, size::Int)
        # When the combination is nonempty, update the count.
        if size > 1
            # Count fixed points: those indices i for which v[i]==i, then subtract 1.
            n_selfroots[size] += count(v .== id) - 1
        end
        # Try to extend the combination.
        for i in (last+1):nm
            # Compute the new composition.
            new_v = state[i].root[v]
            rec(i, new_v, size + 1)
        end
    end
    
    @inbounds for j=1:nm
        rec(j, state[j].root, 1)
    end
    
    # Divide each total by the number of combinations of that size.
    @inbounds for j in 2:nm
        n_selfroots[j] /= binomial(nm, j)
    end
    
    # for only singleton forests, we have already the root_counters:
    n_selfroots[1] = mean(s.root_counter for s in state)
    
    return n_selfroots
end

function find_the_root_to_awaken(q::Float64, state::RfState)
    # function to find the next q and the associated root to awaken
    u = rand(Beta(state.root_counter, 1)) * (q / (q + state.dmax)) # the maximum of root_counter iid uniform variables on [0, q / (q + state.dmax)]. Distrib is rx^{r-1} with r root counter
    q = (u / (1 - u)) * state.dmax # the corresponding q
    r = rand(state.root_set) # uniformly chosen root
    d_r = state.deg[r] # its degree
    while true
        if rand() <  ((state.dmax - d_r) / state.dmax) * (q / (q + d_r)) # if we go through the self loop (left of * sign) AND we die right away (right of * sign), then it is a short awakening
            state.short_awakenings += 1 # counts the number of short awakenings
            #sample again q and r (using the new q!)
            u = rand(Beta(state.root_counter, 1)) * (q / (q + state.dmax))
            q = (u / (1 - u)) * state.dmax
            r = rand(state.root_set)
            d_r = state.deg[r]
        else
            break
        end
    end
    return r,q
end

function find_the_root_to_awaken(q::Float64, state::RfState_PQ)
    # function to find the next q and the associated root to awaken
    return dequeue_pair!(state.list_q)
end

function find_the_root_to_awaken(q::Float64, state::RfState_leaves)
    # function to find the next qstar and the associated root to awaken
    root_set = keys(state.leaf_sets)
    u = rand(Beta(state.root_counter, 1)) * (q / (q + state.dmax)) # distrib is rx^{r-1} with r root counter
    q = (u / (1 - u)) * state.dmax
    r = rand(root_set) # keys(state.leaf_sets) is the list of current roots
    d_r = state.deg[r]
    while true
        #if (rand() < q / (q + d_r)) && (rand() < (state.dmax - d_r) / state.dmax)  # if we go through the self loop AND we die right away, then this is a false awakening
        if rand() <  ((state.dmax - d_r) / state.dmax) * (q / (q + d_r)) # if we go through the self loop AND we die right away, then this is a false awakening
            state.false_awakenings += 1
            u = rand(Beta(state.root_counter, 1)) * (q / (q + state.dmax)) # distrib is rx^{r-1} with r root counter
            q = (u / (1 - u)) * state.dmax
            r = rand(root_set)
            d_r = state.deg[r]
        else
            break
        end
    end
    return r,q
end

function find_the_root_to_awaken(q::Float64, state::RfState_leaves_PQ)
    # function to find the next q and the associated root to awaken
    return dequeue_pair!(state.list_q)
end

function reboot!(g::SimpleGraph{Int64}, q::Float64, r::Int64, state::RfState)
    function look_ahead_no_node_is_alive(state::RfState, i::Int64)
        #outputs the first node down the line of i that is either a root or alive (in the case where we know that no node is alive because we are at the beginnning of reboot!)
        current = state.next[i]
        if state.root_is_correct
            state.root_is_correct = false
            return state.root[current]
        end
        while !state.is_root[current] #run the while loop until current is either alive or root (or both)
            current = state.next[current]
        end
        return current
    end
    function look_ahead(state::RfState, i::Int64)
        #outputs the first node down the line of i that is either a root or alive
        current = state.next[i]
        while true
            if state.is_alive[current] || state.is_root[current] #run the while loop until current is either alive or root (or both)
                break
            end
            current = state.next[current]
        end
        return current
    end
    function add_loop_to_alive_nodes!(state::RfState, i::Int64)
        # Aliven all nodes in the loop in which i belongs
        current = i
        while true
            if state.is_alive[current] #until an alive node is encountered
                break
            end
            push!(state.alive_nodes, current)
            state.is_alive[current] = true
            current = state.next[current]
        end
        return state
    end
    # this function reboots the tree rooted in r
    state.next[r] = rand(neighbors(g, r)) # the first step cannot be 'death'
    node_ahead = look_ahead_no_node_is_alive(state, r) #node_ahead is the first node following the arrows after r that is a root

    # r is no longer a root:
    state.is_root[r] = false 
    delete!(state.root_set, r)
    state.root_counter -= 1

    if node_ahead == r #if crashed into itself. NB: if crashed into other tree: do nothing
        add_loop_to_alive_nodes!(state, r) # Aliven all nodes in the detected loop
        while !isempty(state.alive_nodes)
            i = pop!(state.alive_nodes) #take an arbitrary node to awake
            if rand() < q / (q + state.deg[i]) #dies
                state.is_root[i] = true
                push!(state.root_set, i)
                state.root[i] = i
                state.root_counter += 1
                state.is_alive[i] = false
            else
                state.next[i] = rand(neighbors(g, i))
                node_ahead = look_ahead(state, i) #node_ahead is the first node down the line that is either rooted or alive
                state.is_alive[i] = false
                if node_ahead == i #if crashed into itself. NB: #if crashed into other tree or other branch: do nothing
                    add_loop_to_alive_nodes!(state, i) # Aliven all nodes in the detected loop
                end 
            end
        end
    end
    return state
end

function reboot!(g::SimpleGraph{Int64}, q::Float64, r::Int64, state::RfState_PQ)
    function look_ahead_no_node_is_alive(state::RfState_PQ, i::Int64)
        #outputs the first node down the line of i that is either a root or alive (in the case where we know that no node is alive because we are at the beginnning of reboot!)
        current = state.next[i]
        if state.root_is_correct
            state.root_is_correct = false
            return state.root[current]
        end
        while !state.is_root[current] #run the while loop until current is either alive or root (or both)
            current = state.next[current]
        end
        return current
    end
    function look_ahead(state::RfState_PQ, i::Int64)
        #outputs the first node down the line of i that is either a root or alive
        current = state.next[i]
        while true
            if state.is_alive[current] || state.is_root[current] #run the while loop until current is either alive or root (or both)
                break
            end
            current = state.next[current]
        end
        return current
    end
    function add_loop_to_alive_nodes!(state::RfState_PQ, i::Int64)
        # Aliven all nodes in the loop in which i belongs
        current = i
        while true
            if state.is_alive[current] #until an alive node is encountered
                break
            end
            push!(state.alive_nodes, current)
            state.is_alive[current] = true
            current = state.next[current]
        end
        return state
    end
    # this function reboots the tree rooted in r
    state.next[r] = rand(neighbors(g, r)) # the first step cannot be 'death'
    node_ahead = look_ahead_no_node_is_alive(state, r) #node_ahead is the first node following the arrows after r that is a root

    # r is no longer a root:
    state.is_root[r] = false 
    delete!(state.root_set, r)
    state.root_counter -= 1

    if node_ahead == r #if crashed into itself. NB: if crashed into other tree: do nothing
        add_loop_to_alive_nodes!(state, r) # Aliven all nodes in the detected loop
        while !isempty(state.alive_nodes)
            i = pop!(state.alive_nodes) #take an arbitrary node to awake
            u = rand()
            if u < q / (q + state.deg[i]) #dies
                state.is_root[i] = true
                push!(state.root_set, i)
                state.root[i] = i
                state.root_counter += 1
                enqueue!(state.list_q, i => (u / (1 - u)) * state.deg[i])
                state.is_alive[i] = false
            else
                state.next[i] = rand(neighbors(g, i))
                node_ahead = look_ahead(state, i) #node_ahead is the first node down the line that is either rooted or alive
                state.is_alive[i] = false
                if node_ahead == i #if crashed into itself. NB: #if crashed into other tree or other branch: do nothing
                    add_loop_to_alive_nodes!(state, i) # Aliven all nodes in the detected loop
                end 
            end
        end
    end
    return state
end

function reboot!(g::SimpleGraph{Int64}, q::Float64, r::Int64, state::RfState_leaves)
    function assign_new_root!(state::RfState_leaves, r::Int64)
        new_root = state.root[state.next[r]]
        if state.is_leaf[state.next[r]]
            state.is_leaf[state.next[r]] = false # we should here, in theory, also remove state.next[r] from the list of leaves of new_root: but it is long and it doesn't matter if some leaves of some roots are not actually leaves
        end
        leaf_set_of_r = pop!(state.leaf_sets, r)
        append!(state.leaf_sets[new_root], leaf_set_of_r)
        for leaf in leaf_set_of_r
            current = leaf
            while state.root[current] == r
                state.root[current] = new_root
                current = state.next[current]
            end
        end
    end
    function get_connected_comp_and_unroot!(state::RfState_leaves, r::Int64)
        leaf_set_of_r = pop!(state.leaf_sets, r)
        leaf_set_of_r = leaf_set_of_r[state.is_leaf[leaf_set_of_r]] # this is because true leaves of tree rooted in r are only a subset of state.leaf_sets[r] (see implem of Wilson). I'm not sure we need it
        for leaf in leaf_set_of_r
            current = leaf
            while state.root[current] == r
                state.nextsave[current] = state.next[current]
                state.next_is_already_sampled[current] = true
                state.root[current] = 0
                current = state.next[current]
            end
        end
        return leaf_set_of_r
    end
    # this function reboots the tree rooted in r
    n = nv(g)
    state.next[r] = rand(neighbors(g, r)) # the first step cannot be 'death'
    state.root_counter -= 1
    
    if state.root[state.next[r]] != r # ran into other tree
        assign_new_root!(state, r)
    else #do something more subtle
        leaf_set_of_r = get_connected_comp_and_unroot!(state, r)
        state.nextsave[r] = n+1
        for leaf in leaf_set_of_r
            i = leaf
            while state.nextsave[i] != 0 # while we have not been here yet
                if state.root[i] == 0                     # if state.root[i] == 0
                    current = i
                    while state.root[current] == 0
                        if state.next_is_already_sampled[current]
                            state.next_is_already_sampled[current] = false
                            current = state.next[current]
                        else
                            u = rand()
                            if u < q / (q + state.deg[current]) #dies
                                state.next[current] = n + 1 #n+1 is the death node
                                state.root[current] = current #here's the root!
                                root_of_tree = current
                                state.root_counter += 1
                                push!(state.leaf_sets, root_of_tree => Int64[])
                            else
                                state.next[current] = rand(neighbors(g, current))
                            end
                            current = state.next[current]
                        end
                    end
                    if current != n+1    # if ran into other tree
                        root_of_tree = state.root[current]
                        if state.is_leaf[current]
                            state.is_leaf[current] = false
                        end
                    end
                    state.is_leaf[i] = true
                    push!(state.leaf_sets[root_of_tree], i)
                    current = i
                    while state.root[current] == 0
                        state.root[current] = root_of_tree
                        current = state.next[current]
                    end
                end
                iold = i
                i = state.nextsave[i]
                state.nextsave[iold] = 0
            end
        end
    end
    return state
end

function reboot!(g::SimpleGraph{Int64}, q::Float64, r::Int64, state::RfState_leaves_PQ)
    function assign_new_root!(state::RfState_leaves_PQ, r::Int64)
        new_root = state.root[state.next[r]]
        if state.is_leaf[state.next[r]]
            state.is_leaf[state.next[r]] = false # we should here, in theory, also remove state.next[r] from the list of leaves of new_root: but it is long and it doesn't matter if some leaves of some roots are not actually leaves
        end
        leaf_set_of_r = pop!(state.leaf_sets, r)
        append!(state.leaf_sets[new_root], leaf_set_of_r)
        for leaf in leaf_set_of_r
            current = leaf
            while state.root[current] == r
                state.root[current] = new_root
                current = state.next[current]
            end
        end
    end
    function get_connected_comp_and_unroot!(state::RfState_leaves_PQ, r::Int64)
        leaf_set_of_r = pop!(state.leaf_sets, r)
        leaf_set_of_r = leaf_set_of_r[state.is_leaf[leaf_set_of_r]] # this is because true leaves of tree rooted in r are only a subset of state.leaf_sets[r] (see implem of Wilson). I'm not sure we need it
        for leaf in leaf_set_of_r
            current = leaf
            while state.root[current] == r
                state.nextsave[current] = state.next[current]
                state.next_is_already_sampled[current] = true
                state.root[current] = 0
                current = state.next[current]
            end
        end
        return leaf_set_of_r
    end
    # this function reboots the tree rooted in r
    n = nv(g)
    state.next[r] = rand(neighbors(g, r)) # the first step cannot be 'death'
    state.root_counter -= 1
    
    if state.root[state.next[r]] != r # ran into other tree
        assign_new_root!(state, r)
    else #do something more subtle
        leaf_set_of_r = get_connected_comp_and_unroot!(state, r)
        state.nextsave[r] = n+1
        for leaf in leaf_set_of_r
            i = leaf
            while state.nextsave[i] != 0 # while we have not been here yet
                if state.root[i] == 0                     # if state.root[i] == 0
                    current = i
                    while state.root[current] == 0
                        if state.next_is_already_sampled[current]
                            state.next_is_already_sampled[current] = false
                            current = state.next[current]
                        else
                            u = rand()
                            if u < q / (q + state.deg[current]) #dies
                                enqueue!(state.list_q, current => (u / (1 - u)) * state.deg[current]) #the limit value of q for which it would not have died in current node (given this u)
                                state.next[current] = n + 1 #n+1 is the death node
                                state.root[current] = current #here's the root!
                                root_of_tree = current
                                state.root_counter += 1
                                push!(state.leaf_sets, root_of_tree => Int64[])
                            else
                                state.next[current] = rand(neighbors(g, current))
                            end
                            current = state.next[current]
                        end
                    end
                    if current != n+1    # if ran into other tree
                        root_of_tree = state.root[current]
                        if state.is_leaf[current]
                            state.is_leaf[current] = false
                        end
                    end
                    state.is_leaf[i] = true
                    push!(state.leaf_sets[root_of_tree], i)
                    current = i
                    while state.root[current] == 0
                        state.root[current] = root_of_tree
                        current = state.next[current]
                    end
                end
                iold = i
                i = state.nextsave[i]
                state.nextsave[iold] = 0
            end
        end
    end
    return state
end

function wilson!(g::SimpleGraph{Int64}, q::Float64, state::RfState)
    n = nv(g)
    In_tree = falses(n)

    for i=1:n
        current = i
        while !In_tree[current]
            if rand() < q / (q + state.deg[current]) #dies
                state.is_root[current] = true
                push!(state.root_set, current)
                state.root_counter += 1
                state.root[current] = current
                In_tree[current] = true
            else
                state.next[current] = rand(neighbors(g, current))
                current = state.next[current]
            end
        end
        root_of_tree = state.root[current]
        current = i
        while !In_tree[current]
            In_tree[current] = true
            state.root[current] = root_of_tree #not necessary as we don't take this into account
            current = state.next[current]
        end
    end
    state.root_is_correct = true
    return state
end

function wilson!(g::SimpleGraph{Int64}, q::Float64, state::RfState_PQ)
    n = nv(g)
    In_tree = falses(n)

    for i=1:n
        current = i
        while !In_tree[current]
            u = rand()
            if u < q / (q + state.deg[current]) #dies
                state.is_root[current] = true
                push!(state.root_set, current)
                enqueue!(state.list_q, current => (u / (1 - u)) * state.deg[current]) #the limit value of q for which it would not have died in current node (given this u)
                state.root_counter += 1
                state.root[current] = current
                In_tree[current] = true
            else
                state.next[current] = rand(neighbors(g, current))
                current = state.next[current]
            end
        end
        root_of_tree = state.root[current]
        current = i
        while !In_tree[current]
            In_tree[current] = true
            state.root[current] = root_of_tree #not necessary as we don't take this into account
            current = state.next[current]
        end
    end
    state.root_is_correct = true
    return state
end

function wilson!(g::SimpleGraph{Int64}, q::Float64, state::RfState_leaves)
    n = nv(g)
    state.root[n+1] = n+1
    root_of_tree = 0

    for i=1:n
        if state.root[i] == 0
            current = i
            while state.root[current] == 0
                u = rand()
                if u < q / (q + state.deg[current]) #dies
                    state.next[current] = n + 1 #n+1 is the death node
                    state.root[current] = current #here's the root!
                    root_of_tree = current
                    state.root_counter += 1
                    push!(state.leaf_sets, root_of_tree => Int64[])
                else
                    state.next[current] = rand(neighbors(g, current))
                end
                current = state.next[current]
            end
            if current != n+1    # if ran into other tree
                if state.is_leaf[current]
                    state.is_leaf[current] = false
                    #TODO: if state.is_leaf[current] was true, remove current from
                    # state.leaf_sets[root_of_tree], no?
                    # after checking: no need as state.leaf_sets[root_of_tree] includes the leaf set  but does not need to be equal to it
                end
                root_of_tree = state.root[current]
            end
            state.is_leaf[i] = true
            push!(state.leaf_sets[root_of_tree], i)
            current = i
            while state.root[current] == 0
                state.root[current] = root_of_tree
                current = state.next[current]
            end
        end
    end
    return state
end

function wilson!(g::SimpleGraph{Int64}, q::Float64, state::RfState_leaves_PQ)
    n = nv(g)
    state.root[n+1] = n+1
    root_of_tree = 0

    for i=1:n
        if state.root[i] == 0
            current = i
            while state.root[current] == 0
                u = rand()
                if u < q / (q + state.deg[current]) #dies
                    enqueue!(state.list_q, current => (u / (1 - u)) * state.deg[current]) #the limit value of q for which it would not have died in current node (given this u)
                    state.next[current] = n + 1 #n+1 is the death node
                    state.root[current] = current #here's the root!
                    root_of_tree = current
                    state.root_counter += 1
                    push!(state.leaf_sets, root_of_tree => Int64[])
                else
                    state.next[current] = rand(neighbors(g, current))
                end
                current = state.next[current]
            end
            if current != n+1    # if ran into other tree
                if state.is_leaf[current]
                    state.is_leaf[current] = false
                    #TODO: if state.is_leaf[current] was true, remove current from
                    # state.leaf_sets[root_of_tree], no?
                    # after checking: no need as state.leaf_sets[root_of_tree] includes the leaf set  but does not need to be equal to it
                end
                root_of_tree = state.root[current]
            end
            state.is_leaf[i] = true
            push!(state.leaf_sets[root_of_tree], i)
            current = i
            while state.root[current] == 0
                state.root[current] = root_of_tree
                current = state.next[current]
            end
        end
    end
    return state
end

## test with true moments

function collect_true_moments(true_eigs, qrange :: Vector{Float64}, n_moments::Int64)
    n = length(true_eigs)
    N_q = length(qrange)
    true_moments = zeros(n_moments, N_q)
    for iq in 1:N_q
        for im in 1:n_moments
            true_moments[im, iq] = sum((qrange[iq] ./ (qrange[iq] .+ true_eigs)) .^ im) ./ n
        end
    end
    var_bound = max.((1e-5 * true_moments).^2, 1e-20)
    moments = Vector{@NamedTuple{y::Vector{Float64}, var::Vector{Float64}}}(undef,N_q)
    if n_moments>1
        for i=1:N_q
            moments[i] = (y = true_moments[:,i], var = var_bound[:,i])
        end
    else 
        for i=1:N_q
            moments[i] = (y = [true_moments[i]], var = [var_bound[i]])
        end
    end
    return moments
end

## verify

function test_MC_first_moments(g, MC, algo)
    qmax = Float64(2*maximum(degree(g)))
    qmin = mean(degree(g))/100
    nq = 5
    qrange = 10 .^range(log10(qmin),log10(qmax),nq)

    n_moments = 4

    L = laplacian_matrix(g)
    L = Matrix(L)
    true_eigs, _ = eigen(L);

    true_moments = zeros(n_moments, nq)
    var_estimator_m1 = zeros(nq)
    for iq in 1:nq
        for im in 1:n_moments
            true_moments[im, iq] = sum((qrange[iq] ./ (qrange[iq] .+ true_eigs)) .^ im)
        end
        var_estimator_m1[iq] = sum((qrange[iq] .* true_eigs) ./ ((qrange[iq] .+ true_eigs).^2))
    end

    nroots = zeros(Float64, (n_moments,nq, MC))
    for mc = 1:MC
        nroots[:, :, mc] = coupled_forests(g, qrange, n_moments, algo)
    end

    mc_av = zeros(Float64, (n_moments, nq, MC-1)) #zeros(Float64, (length(qrange), MC-1))
    mc_var = zeros(Float64, (n_moments, nq, MC-1)) #zeros(Float64, (length(qrange), MC-1))
    for i=2:MC
        mc_av[:,:,i-1] = mean(nroots[:,:,1:i], dims=3)
        mc_var[:,:,i-1] = var(nroots[:,:,1:i], dims=3)
    end
    return mc_av, mc_var, true_moments, var_estimator_m1, qrange
end

function verify_coupled_forest_implem(algo::String)
    MC = 500 # number of MC estimates
    
    n = 10^3 #number of nodes
    begin # Erdos Rényi
        p = 3 * log(n) / n
        m = convert(Int64, round(p*n*(n-1)/2))
        g = erdos_renyi(n, m)
    end

    println("Testing on sparse ER graph")
    mc_av, mc_var, true_moments, var_estimator_m1, qrange = test_MC_first_moments(g, MC, algo)
    nq = length(qrange)
    
    begin
        p1 = plot(2:MC, abs.(mc_av[1,:,:])', linewidth=2, label = reshape(["q = $(round(qrange[i], sigdigits=2))" for i=1:nq], 1, nq))
        plot!(p1, xlabel = "# MC", ylabel = "1st m est", yaxis=:log, xticks=[0,floor(MC/2),MC])
        hline!(p1, true_moments[1,:], linewidth=1, linecolor = :black, label=false);

        p2 = plot(2:MC, abs.(mc_var[1,:,:])', linewidth=2, label = reshape(["q = $(round(qrange[i], sigdigits=2))" for i=1:nq], 1, nq));
        plot!(p2, xlabel = "# MC", ylabel = "Var 1st mom", yaxis=:log, xticks=[0,floor(MC/2),MC])
        hline!(p2, var_estimator_m1, linewidth=1, linecolor = :black, label=false);

        p3 = plot(2:MC, abs.(mc_av[2,:,:])', linewidth=2, label = reshape(["q = $(round(qrange[i], sigdigits=2))" for i=1:nq], 1, nq))
        plot!(p3, xlabel = "# MC", ylabel = "2nd m est", yaxis=:log, xticks=[0,floor(MC/2),MC])
        hline!(p3, true_moments[2,:], linewidth=1, linecolor = :black, label=false);

        p4 = plot(2:MC, abs.(mc_av[3,:,:])', linewidth=2, label = reshape(["q = $(round(qrange[i], sigdigits=2))" for i=1:nq], 1, nq))
        plot!(p4, xlabel = "# MC", ylabel = "3rd m est", yaxis=:log, xticks=[0,floor(MC/2),MC])
        hline!(p4, true_moments[3,:], linewidth=1, linecolor = :black, label=false);

        p5 = plot(2:MC, abs.(mc_av[4,:,:])', linewidth=2, label = reshape(["q = $(round(qrange[i], sigdigits=2))" for i=1:nq], 1, nq))
        plot!(p5, xlabel = "# MC", ylabel = "4th m est", yaxis=:log, xticks=[0,floor(MC/2),MC])
        hline!(p5, true_moments[4,:], linewidth=1, linecolor = :black, label=false);
    end

    ###############################################################

    begin # Erdos Rényi
        p = 100 * log(n) / n
        m = convert(Int64, round(p*n*(n-1)/2))
        g = erdos_renyi(n, m)
    end

    println("Testing on dense ER graph")
    mc_av, mc_var, true_moments, var_estimator_m1, qrange = test_MC_first_moments(g, MC, algo)
    nq = length(qrange)
    
    begin
        p6 = plot(2:MC, abs.(mc_av[1,:,:])', linewidth=2, label = reshape(["q = $(round(qrange[i], sigdigits=2))" for i=1:nq], 1, nq))
        plot!(p6, xlabel = "# MC", ylabel = "1st m est", yaxis=:log, xticks=[0,floor(MC/2),MC])
        hline!(p6, true_moments[1,:], linewidth=1, linecolor = :black, label=false);

        p7 = plot(2:MC, abs.(mc_var[1,:,:])', linewidth=2, label = reshape(["q = $(round(qrange[i], sigdigits=2))" for i=1:nq], 1, nq));
        plot!(p7, xlabel = "# MC", ylabel = "Var 1st mom", yaxis=:log, xticks=[0,floor(MC/2),MC])
        hline!(p7, var_estimator_m1, linewidth=1, linecolor = :black, label=false);

        p8 = plot(2:MC, abs.(mc_av[2,:,:])', linewidth=2, label = reshape(["q = $(round(qrange[i], sigdigits=2))" for i=1:nq], 1, nq))
        plot!(p8, xlabel = "# MC", ylabel = "2nd m est", yaxis=:log, xticks=[0,floor(MC/2),MC])
        hline!(p8, true_moments[2,:], linewidth=1, linecolor = :black, label=false);

        p9 = plot(2:MC, abs.(mc_av[3,:,:])', linewidth=2, label = reshape(["q = $(round(qrange[i], sigdigits=2))" for i=1:nq], 1, nq))
        plot!(p9, xlabel = "# MC", ylabel = "3rd m est", yaxis=:log, xticks=[0,floor(MC/2),MC])
        hline!(p9, true_moments[3,:], linewidth=1, linecolor = :black, label=false);

        p10 = plot(2:MC, abs.(mc_av[4,:,:])', linewidth=2, label = reshape(["q = $(round(qrange[i], sigdigits=2))" for i=1:nq], 1, nq))
        plot!(p10, xlabel = "# MC", ylabel = "4th m est", yaxis=:log, xticks=[0,floor(MC/2),MC])
        hline!(p10, true_moments[4,:], linewidth=1, linecolor = :black, label=false);
    end

    ###############################################################

    begin # BA
        g = barabasi_albert(n, 30)
    end
    println("Testing on BA graph")

    mc_av, mc_var, true_moments, var_estimator_m1, qrange = test_MC_first_moments(g, MC, algo)
    nq = length(qrange)
    
    begin
        p11 = plot(2:MC, abs.(mc_av[1,:,:])', linewidth=2, label = reshape(["q = $(round(qrange[i], sigdigits=2))" for i=1:nq], 1, nq))
        plot!(p11, xlabel = "# MC", ylabel = "1st m est", yaxis=:log, xticks=[0,floor(MC/2),MC])
        hline!(p11, true_moments[1,:], linewidth=1, linecolor = :black, label=false);

        p12 = plot(2:MC, abs.(mc_var[1,:,:])', linewidth=2, label = reshape(["q = $(round(qrange[i], sigdigits=2))" for i=1:nq], 1, nq));
        plot!(p12, xlabel = "# MC", ylabel = "Var 1st mom", yaxis=:log, xticks=[0,floor(MC/2),MC])
        hline!(p12, var_estimator_m1, linewidth=1, linecolor = :black, label=false);

        p13 = plot(2:MC, abs.(mc_av[2,:,:])', linewidth=2, label = reshape(["q = $(round(qrange[i], sigdigits=2))" for i=1:nq], 1, nq))
        plot!(p13, xlabel = "# MC", ylabel = "2nd m est", yaxis=:log, xticks=[0,floor(MC/2),MC])
        hline!(p13, true_moments[2,:], linewidth=1, linecolor = :black, label=false);

        p14 = plot(2:MC, abs.(mc_av[3,:,:])', linewidth=2, label = reshape(["q = $(round(qrange[i], sigdigits=2))" for i=1:nq], 1, nq))
        plot!(p14, xlabel = "# MC", ylabel = "3rd m est", yaxis=:log, xticks=[0,floor(MC/2),MC])
        hline!(p14, true_moments[3,:], linewidth=1, linecolor = :black, label=false);

        p15 = plot(2:MC, abs.(mc_av[4,:,:])', linewidth=2, label = reshape(["q = $(round(qrange[i], sigdigits=2))" for i=1:nq], 1, nq))
        plot!(p15, xlabel = "# MC", ylabel = "4th m est", yaxis=:log, xticks=[0,floor(MC/2),MC])
        hline!(p15, true_moments[4,:], linewidth=1, linecolor = :black, label=false);
    end

    ###############################################################

    begin # grid
        g = Graphs.grid([100,Int(n/100)]; periodic=true);
    end

    println("Testing on grid graph")

    mc_av, mc_var, true_moments, var_estimator_m1, qrange = test_MC_first_moments(g, MC, algo)
    nq = length(qrange)
    
    begin
        p16 = plot(2:MC, abs.(mc_av[1,:,:])', linewidth=2, label = reshape(["q = $(round(qrange[i], sigdigits=2))" for i=1:nq], 1, nq))
        plot!(p16, xlabel = "# MC", ylabel = "1st m est", yaxis=:log, xticks=[0,floor(MC/2),MC])
        hline!(p16, true_moments[1,:], linewidth=1, linecolor = :black, label=false);

        p17 = plot(2:MC, abs.(mc_var[1,:,:])', linewidth=2, label = reshape(["q = $(round(qrange[i], sigdigits=2))" for i=1:nq], 1, nq));
        plot!(p17, xlabel = "# MC", ylabel = "Var 1st mom", yaxis=:log, xticks=[0,floor(MC/2),MC])
        hline!(p17, var_estimator_m1, linewidth=1, linecolor = :black, label=false);

        p18 = plot(2:MC, abs.(mc_av[2,:,:])', linewidth=2, label = reshape(["q = $(round(qrange[i], sigdigits=2))" for i=1:nq], 1, nq))
        plot!(p18, xlabel = "# MC", ylabel = "2nd m est", yaxis=:log, xticks=[0,floor(MC/2),MC])
        hline!(p18, true_moments[2,:], linewidth=1, linecolor = :black, label=false);

        p19 = plot(2:MC, abs.(mc_av[3,:,:])', linewidth=2, label = reshape(["q = $(round(qrange[i], sigdigits=2))" for i=1:nq], 1, nq))
        plot!(p19, xlabel = "# MC", ylabel = "3rd m est", yaxis=:log, xticks=[0,floor(MC/2),MC])
        hline!(p19, true_moments[3,:], linewidth=1, linecolor = :black, label=false);

        p20 = plot(2:MC, abs.(mc_av[4,:,:])', linewidth=2, label = reshape(["q = $(round(qrange[i], sigdigits=2))" for i=1:nq], 1, nq))
        plot!(p20, xlabel = "# MC", ylabel = "4th m est", yaxis=:log, xticks=[0,floor(MC/2),MC])
        hline!(p20, true_moments[4,:], linewidth=1, linecolor = :black, label=false);
    end

    plot(p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15, p16, p17, p18, p19, p20, layout=(4,5), yaxis=:log, legend=false, xticks=[0,floor(MC/2),MC])#, size=(2500, 1200))
end

## THE FOLLOWING IS NOT USEFUL FOR COUPLED FORESTS: IT IS JUST FOR COMPARISON WITH CLASSICAL WILSON ("FAKE COUPLED FORESTS")

mutable struct RfState_only_Wilson
    root_counter :: Int64
    root :: Vector{Int64}
    next :: Vector{Int64}
end

function RfState_only_Wilson(g::SimpleGraph{Int64})
    n = nv(g)
    return RfState_only_Wilson(
            0, #root_counter
            zeros(Int64, n), #root vector
            zeros(Int64, n) #next
            )
end

function wilson(g::SimpleGraph{Int64}, q::Float64)
    state = RfState_only_Wilson(g);
    deg = convert.(Float64, degree(g))
    n = nv(g)
    In_tree = falses(n)

    for i=1:n
        current = i
        while !In_tree[current]
            u = rand()
            if u < q / (q + deg[current]) #dies
                In_tree[current] = true
                state.root_counter += 1
                state.root[current] = current
            else
                state.next[current] = rand(neighbors(g, current))
                current = state.next[current]
            end
        end
        root_of_tree = state.root[current]
        current = i
        while !In_tree[current]
            In_tree[current] = true
            state.root[current] = root_of_tree
            current = state.next[current]
        end
    end
    return state.root_counter, state.root
end
