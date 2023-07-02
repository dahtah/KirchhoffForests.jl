using RandomForests,Graphs,SimpleWeightedGraphs,Random,Metrics
using Test
using LinearAlgebra
using PyCall
using StatsBase
#test that the result of random_forest is correct
#ie. it should be a spanning forest, oriented towards the roots
function check_correctness(rf)
    F = SimpleDiGraph(rf)
    roots =collect(rf.roots)
    @test !is_cyclic(F)
    @test all(outdegree(F,roots) .== 0)
    #test that all nodes lead to a root
    for cc in connected_components(F)
        if length(cc)==1
            @test cc[1] in roots
        else
            rc = intersect(cc,roots)
            #there should be a single root per connected component
            @test length(rc)==1
            rc = rc[1]
            bf=bfs_parents(F,rc;dir=-1)[cc]
            @test all(bf .> 0)
        end
    end
end

function subgradient_test(y,mu,sol,G,tol)
    # This function calculates lower & upper bounds of subgradients for a given solution
    del_f = zeros(nv(G),2)

    for node = 1 : nv(G)
        ngbrs = neighbors(G,node)
        del_f[node,:] .+= 2*mu*( sol[node] - y[node])
        for j in ngbrs
            d = (sol[node] - sol[j])
            if(d < -tol)
                del_f[node,:] .-= 1
            elseif(d > tol)
                del_f[node,:] .+= 1
            else
                del_f[node,1] -= 1
                del_f[node,2] += 1
            end
        end
    end
    # Check if 0 ∈ ∂f (f is loss_function)
    for node = 1 : nv(G)
        @test ((del_f[node,1] < 0) && (del_f[node,2] > 0)) || (maximum(abs.(del_f[node,:])) <= tol)
    end
end


const testdir = dirname(@__FILE__)
tests = [
    "basic","weighted","smoothing","aliascomparison",
    "sure",
    "smoothing_eff","cv"
]

@testset "RandomForests" begin
    for t in tests
        tp = joinpath(testdir, "$(t).jl")
        include(tp)
    end
end
