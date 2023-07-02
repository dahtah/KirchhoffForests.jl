"""
    smooth_rf_xbar(g :: AbstractGraph,q :: Number,M::SparseMatrixCSC{Float64,Int64},Y::Vector{Float64};maxiter=Inf,abstol=0.0,reltol=0.0,cov_est=false,verbose=false)
    smooth_rf_xbar(g :: AbstractGraph,q :: Array{Number},M::SparseMatrixCSC{Float64,Int64},Y::Vector{Float64};maxiter=Inf,abstol=0.0,reltol=0.0,verbose=false)

Memory/time efficient implementations of the RSF-based estimator ``\\bar{\\mathsf{x}}`` for [GTR](./gtr.md) problem.

# Arguments
- ```g```: Input graph
- ```q```: Forest parameter ``q``. If it is a scalar, it is the same for every vertex in ```g```.
- ```M```: Sparse matrix which equals to ``(\\mathsf{L}/q +\\mathsf{I})`` or ``(\\mathsf{Q}^{-1}\\mathsf{L}+\\mathsf{I})``. Only needed for the residual computation.
- ```Y```: Input graph signal. One can pass a matrix in size of ``n\\times k`` instead of a vector to smooth ``k`` graph signals.

# Optional parameters:
- ```maxiter```: Number of forests to sample
- ```abstol```: Tolerance for the residual ``||\\mathsf{M}*\\mathsf{Y} - \\bar{\\mathbf{x}}||_2``. When the residual is less than the tolerance, the algorithm stops.
- ```reltol```: Stopping threshold for the relative residual ``||\\mathsf{M}*\\mathsf{Y} - \\bar{\\mathbf{x}}||_2 / ||\\mathsf{Y}||_2``. When the relative residual is less than the tolerance, the algorithm stops.
- ```verbose```: Boolean for details. When ```true```, residual error per iteration is reported.

# Outputs 
- ```est```: The estimate given by ``\\bar{\\mathsf{x}}``
- ```indr```: Number of iterations taken  

!!! note "Stopping Criterion"
    When both ```abstol``` and ```reltol``` are specified, the one that yields a larger tolerance is selected as a stopping criterion.
"""
function smooth_rf_xbar(g :: AbstractGraph,q :: Number,M::SparseMatrixCSC{Float64,Int64},Y::Vector{Float64};maxiter=Inf,abstol=0.0,reltol=0.0,cov_est=false,verbose=false)

### SCALARS
    indr = 0.0
    n = nv(g)
### VECTORS
    xbar = zeros(size(Y));
    second_moment = spzeros(n,n);
    cov_mat = spzeros(n,n);
    est = zeros(Float64,n)
    psize = zeros(Float64,n)
    next = zeros(Int64,n)
    root = zeros(Int64,n)
    res_vec = zeros(Float64,n)
### RANDOMIZATION OBJECTS
    rng = MersenneTwister();
    rand_array_size = 2*ne(g)
    p = zeros(rand_array_size)

    rand!(rng,p)
    counter = 1
    normy = norm(Y)
    d = degree(g)
    dq = d .+ q
    randneighbor::Int = 0
    rn::Float64 = 0.0

### SET TOLERANCE
    residual = norm(Y)
    tolerance = max(reltol * residual, abstol)

    while(indr < maxiter)
        root .= Int64(0)
        indr += 1.0

## Forest sampling and calculate xbar
        @inbounds for i in 1:n
            u = Int64(i)
            while (root[u] === 0)
                if(counter === rand_array_size)
                    rand!(rng,p)
                    counter = 1
                end
                rn = p[counter]*dq[u]
                randneighbor = unsafe_trunc(Int64,rn)
                randneighbor += 1
                counter += 1
                if (d[u] < randneighbor)
                    root[u] = u
                    next[u] = 0
                    est[u] = Y[u]
                    psize[u] = 1.0
                else
                    next[u] = g.fadjlist[u][randneighbor]
                    u = next[u]
                end
            end
            r = root[u]
            # Retrace steps, erasing loops
            u = Int64(i)
            while (root[u] === 0)
                est[r] += Y[u]
                psize[r] += 1.0
                root[u] = r
                u = next[u]
            end
        end
        # Compute the estimation
        est ./= psize
        @inbounds @simd for i = 1 : n
            est[i] = est[root[i]]
        end
        axpy!(1.0,est,xbar)
        if(cov_est)
            mul!(second_moment,est,Array(est'),1.0,1.0)
        end
        ## Calculate the residual
        if(tolerance != 0.0)
            mul!(res_vec,M,xbar,1.0/indr,0)
            mul!(res_vec,I,Y,1.0,-1.0)
            if(norm(res_vec) <= tolerance)
                break
            end
        end
    end
    xbar /= indr
    if(cov_est)
        cov_mat = ( second_moment - xbar*(xbar') ./ indr ) ./ (indr-1)
    end
    if(verbose)
        display("Iterations taken:$indr, Residual error: $(norm(res_vec))")
    end
    (est=xbar,cov_mat=cov_mat,nrep=Int(indr))
end


function smooth_rf_xbar(g :: AbstractGraph,q :: Array{Number},M::SparseMatrixCSC{Float64,Int64},Y::Vector{Float64};maxiter=Inf,abstol=0.0,reltol=0.0,verbose=false)

### SCALARS
    indr = 0.0
    n = nv(g)
### VECTORS
    xbar = zeros(size(Y));
    est = zeros(Float64,n)
    psize = zeros(Float64,n)
    next = zeros(Int64,n)
    root = zeros(Int64,n)
    res_vec = zeros(Float64,n)
### RANDOMIZATION OBJECTS
    rng = MersenneTwister();
    rand_array_size = 2*ne(g)
    p = zeros(rand_array_size)

    rand!(rng,p)
    counter = 1
    normy = norm(Y)
    d = degree(g)
    dq = d .+ q
    randneighbor::Int = 0
    rn::Float64 = 0.0

### SET TOLERANCE
    residual = norm(Y)
    tolerance = max(reltol * residual, abstol)

    while(indr < maxiter)
        root .= Int64(0)
        indr += 1.0

## Forest sampling and calculate xbar
        @inbounds for i in 1:n
            u = Int64(i)
            while (root[u] === 0)
                if(counter === rand_array_size)
                    rand!(rng,p)
                    counter = 1
                end
                rn = p[counter]*dq[u]
                randneighbor = unsafe_trunc(Int64,rn)
                randneighbor += 1
                counter += 1
                if (d[u] < randneighbor)
                    root[u] = u
                    next[u] = 0
                    est[u] =  q[u]*Y[u]
                    psize[u] = q[u]
                else
                    next[u] = g.fadjlist[u][randneighbor]
                    u = next[u]
                end
            end
            r = root[u]
            # Retrace steps, erasing loops
            u = Int64(i)
            while (root[u] === 0)
                est[r] += q[u]*Y[u]
                psize[r] += q[u]
                root[u] = r
                u = next[u]
            end
        end
## Compute the estimation
        est ./= psize
        @inbounds @simd for i = 1 : n
            est[i] = est[root[i]]
        end
        axpy!(1.0,est,xbar)
## Calculate the residual
        if(tolerance != 0.0)
            mul!(res_vec,M,xbar,1.0/indr,0)
            mul!(res_vec,I,Y,1.0,-1.0)
            if(norm(res_vec) <= tolerance)
                break
            end
        end
    end
    xbar /= indr
    if(verbose)
        display("Iterations taken:$indr, Residual error: $(norm(res_vec))")
    end
    (est=xbar,nrep=Int(indr))
end

"""
    smooth_rf_xtilde(g :: AbstractGraph,q :: Number,M::SparseMatrixCSC{Float64,Int64},Y::Vector{Float64};maxiter=Inf,abstol=0.0,reltol=0.0,cov_est=false,verbose=false)
    smooth_rf_xtilde(g :: AbstractGraph,q :: Array{Number},M::SparseMatrixCSC{Float64,Int64},Y::Vector{Float64};maxiter=Inf,abstol=0.0,reltol=0.0,verbose=false)

Memory/time efficient implementations of the RSF-based estimator ``\\tilde{\\mathsf{x}}`` for [GTR](./gtr.md) problem.

# Arguments
- ```g```: Input graph
- ```q```: Forest parameter ``q``. If it is a scalar, it is the same for every vertex in ```g```.
- ```M```: Sparse matrix which equals to ``(\\mathsf{L}/q +\\mathsf{I})`` or ``(\\mathsf{Q}^{-1}\\mathsf{L}+\\mathsf{I})``. Only needed for the residual computation.
- ```Y```: Input graph signal. One can pass a matrix in size of ``n\\times k`` instead of a vector to smooth ``k`` graph signals.

# Optional parameters:
- ```maxiter```: Number of forests to sample
- ```abstol```: Tolerance for the residual ``||\\mathsf{M}*\\mathsf{Y} - \\tilde{\\mathbf{x}}||_2``. When the residual is less than the tolerance, the algorithm stops.
- ```reltol```: Stopping threshold for the relative residual ``||\\mathsf{M}*\\mathsf{Y} - \\tilde{\\mathbf{x}}||_2 / ||\\mathsf{Y}||_2``. When the relative residual is less than the tolerance, the algorithm stops.
- ```verbose```: Boolean for details. When ```true```, residual error per iteration is reported.

# Outputs 
- ```est```: The estimate given by ``\\tilde{\\mathsf{x}}``
- ```indr```: Number of iterations taken  

!!! note "Stopping Criterion"
    When both ```abstol``` and ```reltol``` are specified, the one that yields a larger tolerance is selected as a stopping criteria.
"""
function smooth_rf_xtilde(g :: AbstractGraph,q :: Number,M::SparseMatrixCSC{Float64,Int64},Y::Vector{Float64};maxiter=Inf,abstol=0.0,reltol=0.0,verbose=false)

### SCALARS
    indr = 0.0
    n = nv(g)
### VECTORS
    xtilde = zeros(size(Y));
    est = zeros(Float64,n)
    next = zeros(Int64,n)
    root = zeros(Int64,n)
    res_vec = zeros(Float64,n)
### RANDOMIZATION OBJECTS
    rng = MersenneTwister();
    rand_array_size = 2*ne(g)
    p = zeros(rand_array_size)

    rand!(rng,p)
    counter = 1
    normy = norm(Y)
    d = degree(g)
    dq = d .+ q
    randneighbor::Int = 0
    rn::Float64 = 0.0

### SET TOLERANCE
    residual = norm(Y)
    tolerance = max(reltol * residual, abstol)

    while(indr < maxiter)
        root .= Int64(0)
        indr += 1.0

## Forest sampling and calculate xbar
        @inbounds for i in 1:n
            u = Int64(i)
            while (root[u] === 0)
                if(counter === rand_array_size)
                    rand!(rng,p)
                    counter = 1
                end
                rn = p[counter]*dq[u]
                randneighbor = unsafe_trunc(Int64,rn)
                randneighbor += 1
                counter += 1
                if (d[u] < randneighbor)
                    root[u] = u
                    next[u] = 0
                    est[u] = Y[u]
                else
                    next[u] = g.fadjlist[u][randneighbor]
                    u = next[u]
                end
            end
            r = root[u]
            # Retrace steps, erasing loops
            u = Int64(i)
            while (root[u] === 0)
                est[u] = Y[r]
                root[u] = r
                u = next[u]
            end
        end
## Calculate the residual

        axpy!(1.0,est,xtilde)
        if(tolerance != 0.0)
            mul!(res_vec,M,xtilde,1.0/indr,0)
            mul!(res_vec,I,Y,1.0,-1.0)
            if(norm(res_vec) <= tolerance)
                break
            end
        end
    end
    xtilde /= indr
    if(verbose)
        display("Iterations taken:$indr, Residual error: $(norm(res_vec))")
    end
    (est=xtilde,nrep=Int(indr))
end
function smooth_rf_xtilde(g :: AbstractGraph,q :: Array{Number},M::SparseMatrixCSC{Float64,Int64},Y::Vector{Float64};maxiter=Inf,abstol=0.0,reltol=0.0,verbose=false)

### SCALARS
    indr = 0.0
    n = nv(g)
### VECTORS
    xtilde = zeros(size(Y));
    est = zeros(Float64,n)
    next = zeros(Int64,n)
    root = zeros(Int64,n)
    res_vec = zeros(Float64,n)
### RANDOMIZATION OBJECTS
    rng = MersenneTwister();
    rand_array_size = 2*ne(g)
    p = zeros(rand_array_size)

    rand!(rng,p)
    counter = 1
    normy = norm(Y)
    d = degree(g)
    dq = d .+ q
    randneighbor::Int = 0
    rn::Float64 = 0.0

### SET TOLERANCE
    residual = norm(Y)
    tolerance = max(reltol * residual, abstol)

    while(indr < maxiter)
        root .= Int64(0)
        indr += 1.0

## Forest sampling and calculate xbar
        @inbounds for i in 1:n
            u = Int64(i)
            while (root[u] === 0)
                if(counter === rand_array_size)
                    rand!(rng,p)
                    counter = 1
                end
                rn = p[counter]*dq[u]
                randneighbor = unsafe_trunc(Int64,rn)
                randneighbor += 1
                counter += 1
                if (d[u] < randneighbor)
                    root[u] = u
                    next[u] = 0
                    est[u] = Y[u]
                else
                    next[u] = g.fadjlist[u][randneighbor]
                    u = next[u]
                end
            end
            r = root[u]
            # Retrace steps, erasing loops
            u = Int64(i)
            while (root[u] === 0)
                est[u] = Y[r]
                root[u] = r
                u = next[u]
            end
        end
## Calculate the residual

        axpy!(1.0,est,xtilde)
        if(tolerance != 0.0)
            mul!(res_vec,M,xtilde,1.0/indr,0)
            mul!(res_vec,I,Y,1.0,-1.0)
            if(norm(res_vec) <= tolerance)
                break
            end
        end
    end
    xtilde /= indr
    if(verbose)
        display("Iterations taken:$indr, Residual error: $(norm(res_vec))")
    end
    (est=xtilde,nrep=Int(indr))
end
