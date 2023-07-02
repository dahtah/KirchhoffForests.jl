@testset "smoothing_eff" begin

    q = rand()
    g = erdos_renyi(100,0.1)
    L = laplacian_matrix(g)
    M = (L./q + I) # Inverted kernel (needed for residual calculation)
    y = rand(nv(g)) # Graph signal
    eps = 10^-3

    K = inv(Matrix(M))
    xhat = K*y

    ## Check if the second moment of the estimators are the same with the theory
    weighted_varsum_rf_tilde = y'*(I - K'*K)*y
    weighted_varsum_rf_bar = y'*(K - K'*K)*y

    NiterRange = unique(round.(Int64,10 .^ range(0,stop=2,length=10)))
    se_rf_tilde = zeros(length(NiterRange))
    se_rf_bar = zeros(length(NiterRange))
    exp_rep = 10000
    x_rf = zeros(nv(g))
    for (i,nrep) in enumerate(NiterRange)
        for j = 1 : exp_rep
            x_rf = smooth_rf_xtilde(g,q,M,y;maxiter=nrep,abstol=0.0,reltol=0.0).est
            se_rf_tilde[i] += norm(x_rf-xhat)^2
            x_rf = smooth_rf_xbar(g,q,M,y;maxiter=nrep,abstol=0.0,reltol=0.0).est
            se_rf_bar[i] += norm(x_rf-xhat)^2
        end
    end
    se_rf_tilde ./= exp_rep
    se_rf_bar ./= exp_rep
    @test(mse(se_rf_tilde,weighted_varsum_rf_tilde./NiterRange) <= (eps*weighted_varsum_rf_tilde))
    @test(mse(se_rf_bar,weighted_varsum_rf_bar./NiterRange) <= (eps*weighted_varsum_rf_bar))

end
