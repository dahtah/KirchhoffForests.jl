using KirchoffForests,Graphs,SimpleWeightedGraphs,SparseArrays,Random,Metrics
using Test
using LinearAlgebra
using PyCall
using StatsBase


#test that the result of random_forest is correct, i.e., it should be a spanning forest, oriented towards the roots
function check_correctness(rf)
    F = SimpleDiGraph(rf)
    roots = collect(rf.roots)
    @test !is_cyclic(F)
    @test all(outdegree(F, roots) .== 0)
    #test that all nodes lead to a root
    for cc in connected_components(F)
        if length(cc) == 1
            @test cc[1] in roots
        else
            rc = intersect(cc, roots)
            #there should be a single root per connected component
            @test length(rc) == 1
            rc = rc[1]
            bf = bfs_parents(F, rc; dir=-1)[cc]
            @test all(bf .> 0)
        end
    end
end

const testdir = dirname(@__FILE__)
tests = ["basic", "weighted", "smoothing", "aliascomparison","cv","smoothing_eff"]
@testset "KirchoffForests" begin
    for t in tests
        tp = joinpath(testdir, "$(t).jl")
        include(tp)
    end
end
