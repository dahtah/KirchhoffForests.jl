
function approximatelinesearch(y,t_k,mu,L,update;β=0.5)
    α = 1.0
    lossgrad = mu*exp.(t_k) .- mu*y + (L*t_k)
    while ( newton_loss(y,t_k-α*update,mu,L) >  newton_loss(y,t_k,mu,L) .- 0.5*α*lossgrad'*update )
        α = β*α;
    end
    return α
end
function newton_loss(y,t_k,mu,L)
    return -mu*(y'*t_k - sum(exp.(t_k)) ) .+ 0.5*(t_k'*L*t_k)
end

function logfact(y)
    res = 0
    for i = 1 : y
        res += log(i)
    end
    res
end

"""
    newton_poisson_noise(g::AbstractGraph,y::AbstractVector,t0::AbstractVector,mu::Number;α=0.1,numofiter = 100,tol=0.001, method="exact",nrep=100,status=true,line_search=true)
Implements exact and forest updates for solving the problem of Poisson denoising via Laplacian regularization by Newton's method. See [the notebook](https://gricad-gitlab.univ-grenoble-alpes.fr/barthesi/RandomForests.jl/-/blob/docs/docs/src/notebooks/Newton's%20method%20for%20Poisson%20noise/Newton's%20method%20for%20Poisson%20noise.md) for an illustration.

# Arguments
- ```g```: Input Graph 
- ```y```: Input signal 
- ```t0```: Initial solution 
- ```mu```: Regularization parameter 

# Optional Parameters 
- ```α```: Step size for Newton's update
- ```tol```: Tolerance parameter for the increment. Algorithm terminates if ``||\\mathbf{t}_{k+1} -\\mathbf{t}_k||_2< ```tol``` ``  
- ```method```: Method to compute the updates. 
    - ```method="exact"```: Computes the update directly. 
    - ```method="xtilde"```: Computes the update by using ``\\tilde{x}`` 
    - ```method="xbar"```: Computes the update by using ``\\bar{x}``     
- ```numofiter```: Maximum number of iterations for Newton's method
- ```nrep```: Number of spanning forests to sample 
- ```status```: Verbose parameter. 
- ```line_search```: If true, implements approximate line search for the step size α.

# Outputs 
- ```sol```: Solution computed by Newton's iteration
- ```inc_arr```: An array that contains the increment ``||\\mathbf{t}_{k+1} -\\mathbf{t}_k||_2 `` through the iterations. 
- ```loss_arr```: An array that contains the value of loss function through the iterations. 
"""
function newton_poisson_noise(g::AbstractGraph,y::AbstractVector,t0::AbstractVector,mu::Number;α=0.1,numofiter = 100,tol=0.001, method="exact",nrep=100,status=true,line_search=true)
    t_k = copy(t0)
    tprev = copy(t0)
    increment = norm(t0)
    inc_arr = []

    loss_arr = []
    L = laplacian_matrix(g)
    k = 0

    while( increment > tol && k < numofiter)
        vec_k = 1.0 .- (y ./ exp.(t_k)) + ((L*t_k)./(mu .* exp.(t_k)))
        q =  (mu .* exp.(t_k))
        tprev = copy(t_k)
        if (method=="exact")
            update = (smooth(g,q,vec_k))
        else
            vecbar = repeat(mean(vec_k, dims=1), outer=size(y,1))
            Δ = -vecbar + vec_k
            if (method=="xtilde")
                update = (vecbar + smooth_rf(g,q,Δ,[];nrep=nrep,variant=1).est)
            elseif (method=="xbar")
                update = (vecbar + smooth_rf(g,q,Δ,[];nrep=nrep,variant=2).est)
            end
        end
        if(status == true)
            println("Iteration=$k, ||t_k - t_{k+1}||^2=$increment, alpha=$α")
        end

        if(line_search == true)
            α = approximatelinesearch(y,t_k,mu,L,update;β=0.5)
        end
        t_k -= α*update
        k += 1

        increment = norm(tprev - t_k)
        append!(inc_arr,increment)
        loss = newton_loss(y,t_k,mu,L)
        append!(loss_arr,loss)

    end
    println("Method: $method. Terminated after $k iterations, increment $increment")
    sol = exp.(t_k)
    return sol,inc_arr,loss_arr
end

function irls(G,y,z0,mu;numofiter = 100,tol=0.001, method="exact",nrep=100,status=true)
    B = incidence_matrix(G,oriented=true)
    k = 0
    increment = norm(z0)
    z_k = copy(z0)
    z_prev = copy(z0)
    mu = 2*mu
    divbyzerothreshold = (10^(-5))*ones(ne(G))
    loss = zeros(numofiter)
    while( increment > tol && k < numofiter  )
        zprev = copy(z_k)

        grad = abs.(B'*z_k)
        grad .= max.(grad,divbyzerothreshold)
        update = grad .^ (-1)

        M_k = spdiagm(0 => (update))


        L_k = B*M_k*(B')
        L_k = abs.(L_k)
        L_k[diagind(L_k)] .= 0
        G = SimpleWeightedGraph(L_k)

        if (method=="exact")
            z_k = (smooth(G,mu,y))
        else
            ybar = repeat(mean(y, dims=1), outer=size(y,1))
            Δ = -ybar + y
            if (method=="xtilde")
                z_k = (ybar + smooth_rf(G,mu,Δ,[];nrep=nrep,variant=1).est)
            elseif (method=="xbar")
                z_k = (ybar + smooth_rf(G,mu,Δ,[];nrep=nrep,variant=2).est)
            end
        end
        k += 1
        increment = norm(zprev - z_k)
        loss[k] = (mu/2)*norm(y-z_k)^2 + sum(abs.(B'*z_k))
        if(status == true)
            display("Iteration $k, increment $increment, $(loss[k])")
        end
    end
    display("Method: $method. Terminated after $k iterations, increment $increment")
    return z_k,loss
end

"""
    admm_edge_lasso(g::AbstractGraph,q::Real,y::AbstractArray;maxiter=100,ρ=0.001,method="exact",Nfor=20)

Implements exact and forest updates of the ADMM algorithm for solving the problem of edge LASSO. See [the notebook](https://gricad-gitlab.univ-grenoble-alpes.fr/barthesi/RandomForests.jl/-/blob/docs/docs/src/notebooks/Edge%20Lasso-%20ADMM/Edge%20Lasso-%20ADMM.md) for an example. 

# Arguments
- ```g```: Input graph
- ```q```: Regularization parameter 
- ```y```: Input signal

# Optional Parameters 
- ```maxiter```: Maximum number of iterations for ADMM  
- ```α```: Step size for ADMM updates. 
- ```method```: Method to compute the updates. 
    - ```method="exact"```: Computes the update directly. 
    - ```method="xtilde"```: Computes the update by using ``\\tilde{x}`` 
    - ```method="xbar"```: Computes the update by using ``\\bar{x}``     
- ```Nfor```: Number of spanning forests to sample 

# Outputs 
- ```xk```: Solution computed by Newton's iteration
- ```loss_func```: An array that contains the value of the loss function through the iterations. 
"""
function admm_edge_lasso(g::AbstractGraph,q::Real,y::AbstractArray;maxiter=100,ρ=0.001,method="exact",Nfor=20)
    n = nv(g)
    m = ne(g)
    xk = zeros(n)
    zk = zeros(m)
    uk = zeros(m)
    B = incidence_matrix(g,oriented=true)
    L = laplacian_matrix(g)
    loss_func = zeros(maxiter)
    for k = 1 : maxiter
        if(method=="exact")
            xk = (L + (q/ρ)*I)\((q/ρ)*y + B*(zk-uk))
        elseif(method=="xtilde")
            xk = smooth_rf(g,(q/ρ),(y + (ρ/q)*(B*(zk-uk)));nrep=Nfor,variant=1).est
        elseif(method=="xbar")
            xk = smooth_rf(g,(q/ρ),(y + (ρ/q)*(B*(zk-uk)));nrep=Nfor,variant=2).est
        end
        zk = sign.(B'*xk + uk) .* max.(0 , abs.(B'*xk + uk) .- 1/ρ)
        uk += (B'*xk - zk)
        loss_func[k] = 0.5*norm(xk - y)^2 + norm(B'*xk,1)
    end
    xk,loss_func
end
