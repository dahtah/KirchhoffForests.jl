

function newton(G,y,t0,mu;α=0.001,numofiter = 100,tol=0.001, method="exact",nrep=100)
    t_k = copy(t0)
    tprev = copy(t0)
    increment = norm(t0)
    L = laplacian_matrix(G)
    k = 0

    while( increment > tol && k < numofiter)
        vec_k = mu .* exp.(t_k) - mu .* y + L*t_k
        tprev = copy(t_k)
        println("Iteration $k, increment $increment")

        if (method=="exact")
            t_k -= α*(smooth(G,mu .* exp.(t_k),vec_k))
        else
            vecbar = repeat(mean(vec_k, dims=1), outer=size(y,1))
            Δ = -vecbar + vec_k
            if (method=="xtilde")
                t_k -= α*(vecbar + smooth_rf(G,1 ./ (mu .* exp.(t_k)),Δ,[];nrep=nrep,variant=1).est)
            elseif (method=="xbar")
                t_k -= α*(vecbar + smooth_rf(G,1 ./ (mu .* exp.(t_k)),Δ,[];nrep=nrep,variant=2).est)
            end
        end
        k += 1
        increment = norm(tprev - t_k)
    end
    println("Method: $method. Terminated after $k iterations, increment $increment")
    return exp.(t_k)
end

function irls(G,y,z0,mu;numofiter = 100,tol=0.001, method="exact",nrep=100)
    B = incidence_matrix(G,oriented=true)
    k = 0
    increment = norm(z0)
    z_k = copy(z0)
    z_prev = copy(z0)
    while( increment > tol && k < numofiter)
        zprev = copy(z_k)
        M_k = spdiagm(0 => (B'*z_k).^(-1))
        L_k = B*M_k*(B')
        L_k[diagind(L_k)] .= 0
        G = SimpleGraph(-(L_k))
        println("Iteration $k, increment $increment")

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
