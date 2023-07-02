@testset "forest" begin
    Random.seed!(1)
    n = 1000
    gg = LightGraphs.barabasi_albert(n,10)
    for i in 1:100
        rf,bt = random_forest_boundary_track(gg,i)
        check_correctness(rf)
        check_boundary_track(gg,rf,bt)

        rf,bt = random_forest_root_boundary_track(gg,i)
        check_correctness(rf)
        check_root_boundary_track(gg,rf,bt)

        d = LightGraphs.degree(gg)
        prob_vec = (i) ./ (i .+ d) ## Edge probabilities\
        μ = sum(prob_vec)
        σ = sum(prob_vec .* (1 .- prob_vec))^(1/2)
        b = Normal(μ,σ)
        strat_f =  quantile.(b,collect(0.2:0.2:0.8))
        n_strats = length(strat_f)+1
        strats_p = repeat([0.2],n_strats)
        strat_bins = separator2bins(strat_f,n) ## Strat bins are calculated

        fl_roots = cond_bernouilli_sampler(prob_vec,strat_bins[rand(1:5),:])
        rf = random_forest_cond_first_layer(gg,i,BitSet(fl_roots))
        check_correctness(rf)
        check_ss_forests(rf,fl_roots)
    end
end
