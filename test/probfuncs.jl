@testset "probfuncs" begin
    Random.seed!(1)
    n = 1000
    for i in 1 : 1000
        prob_vec = rand(n)
        μ = sum(prob_vec)
        σ = sum(prob_vec .* (1 .- prob_vec))^(1/2)
        b = Normal(μ,σ)
        strat_f =  quantile.(b,collect(0.2:0.2:0.8))
        n_strats = length(strat_f)+1
        strats_p = repeat([0.2],n_strats)
        strat_bins = separator2bins(strat_f,n) ## Strat bins are calculated
        for binidx = 1 : n_strats
            bin =  strat_bins[binidx,:]
            fl_roots = cond_bernouilli_sampler(prob_vec,bin)
            @test  bin[1] < length(fl_roots) <= bin[2]
        end
    end
end
