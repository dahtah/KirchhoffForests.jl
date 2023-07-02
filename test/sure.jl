@testset "sure" begin

    eps_err = 10^-3

    G = erdos_renyi(100,0.1)
    N = nv(G)
    x = rand(N)

    SIGMA = 0.1:0.02:0.2
    MU = round.(10 .^ (-2:1.0:5),sigdigits=2)
    for σ in SIGMA
        exprep = 1000
        risk = zeros(length(MU))
        surescore_hat = zeros(length(MU))

        for i = 1 : exprep
            y = x .+ randn(length(x))*σ
            nrepsure = 10
            mu_hat,scores_hat = SURE(G,y;σ=σ,mu_range=MU,nrep = nrepsure,method="exact")

            surescore_hat += scores_hat
            for (i,mu) in enumerate(MU)
                xhat = smooth(G,mu,y)
                risk[i] += norm(xhat - x)^2
            end

        end
        risk ./= exprep
        surescore_hat ./= exprep
        @test (norm(surescore_hat-risk)^2/length(SIGMA)) <= eps_err
    end
end
