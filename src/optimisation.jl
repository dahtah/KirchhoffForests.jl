
function newton(G,y,t0,mu;α=0.1,numofiter = 100,tol=0.001, method="exact",nrep=100,status=true,line_search=true)
    t_k = copy(t0)
    tprev = copy(t0)
    increment = norm(t0)
    L = laplacian_matrix(G)
    k = 0

    while( increment > tol && k < numofiter)
        vec_k = 1.0 .- (y ./ exp.(t_k)) + ((L*t_k)./(mu .* exp.(t_k)))
        q =  (mu .* exp.(t_k))
        tprev = copy(t_k)
        if (method=="exact")
            update = (smooth(G,q,vec_k))
        else
            vecbar = repeat(mean(vec_k, dims=1), outer=size(y,1))
            Δ = -vecbar + vec_k
            if (method=="xtilde")
                update = (vecbar + smooth_rf(G,q,Δ,[];nrep=nrep,variant=1).est)
            elseif (method=="xbar")
                update = (vecbar + smooth_rf(G,q,Δ,[];nrep=nrep,variant=2).est)
            end
        end
        if(status == true)
            println("Iteration $k, increment $increment, alpha $α")
        end

        if(line_search == true)
            α = approximatelinesearch(y,t_k,mu,L,update;β=0.5)
        end
        t_k -= α*update
        k += 1
        increment = norm(tprev - t_k)
    end
    println("Method: $method. Terminated after $k iterations, |zk-zk+1| increment $increment")
    return exp.(t_k)
end
function approximatelinesearch(y,t_k,mu,L,update;β=0.5)
    α = 1.0
    lossgrad = mu*exp.(t_k) .- mu*y + (L*t_k)
    while ( newton_loss(y,t_k-α*update,mu,L) >  newton_loss(y,t_k,mu,L) .- 0.5*α*lossgrad'*update )
        α = β*α;
    end
    return α
end
function newton_loss(y,t_k,mu,L)
    return -mu*(y'*t_k - sum(exp.(t_k)) .- sum(logfact.(y)) ) .+ 0.5*(t_k'*L*t_k)
end

function logfact(y)
    res = 0
    for i = 1 : y
        res += log(i)
    end
    res
end
function irls(G,y,z0,mu;numofiter = 100,tol=0.001, method="exact",nrep=100,status=true)
    B = incidence_matrix(G,oriented=true)
    k = 0
    increment = norm(z0)
    z_k = copy(z0)
    z_prev = copy(z0)
    while( increment > tol && k < numofiter  )
        zprev = copy(z_k)
        update =(abs.(B'*z_k) .+ 0.0001).^(-1)
        M_k = spdiagm(0 => (update))

        if(status == true)
            println("Iteration $k, increment $increment")
        end

        L_k = B*M_k*(B')
        L_k[diagind(L_k)] .= 0
        G = SimpleWeightedGraph(-(L_k + L_k')./2)

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
    end
    println("Method: $method. Terminated after $k iterations, increment $increment")
    return z_k
end
