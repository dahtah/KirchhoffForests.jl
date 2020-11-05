function SURE(G,Y;σ=0.1,mu_range=0.1:0.1:1.0,nrep = 10,method="xbar")
    min_score = typemax(Float64)
    minmu = 0.0
    score =zeros(length(mu_range))
    for (i,mmu) in enumerate(mu_range)

        if (method=="exact")
            Id = spdiagm(0 => ones(nv(G)))
            f = smooth(G,mmu,Y)
            nroot = sum(diag(smooth(G,mmu,Id)))
        else
            Ybar = repeat(mean(Y, dims=1), outer=size(Y,1))
            Δ = Y - Ybar
            if (method=="xtilde")
                f,nroot,_ = smooth_rf(G,mmu,Δ,[];variant=1,nrep=nrep)
                f = (Ybar + f)
            elseif (method=="xbar")
                f,nroot,_ = smooth_rf(G,mmu,Δ,[];variant=2,nrep=nrep)
                f = (Ybar + f)
            end
        end
        score[i] = norm(Y .- f)^2 + σ^2*(-nv(G) + 2*nroot)
        if(min_score > score[i])
            min_score = score[i]
            minmu = mmu
        end
    end
    return minmu,score
end
