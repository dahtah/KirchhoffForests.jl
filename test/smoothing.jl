@testset "smoothing" begin
#=
TODO:
- describe the test
- add "using/import"s
=#
    g=grid([11], periodic=true)
    #t = LinRange(0,1,10)
    t = (0:10) ./ 11
    y = cos.(2*pi*t)
    ys = smooth(g, .1, y)
    @assert (ys / maximum(ys)) ≈ y  #should be eigenvector
    ysw = smooth(SimpleWeightedGraph(g), .1, y)
    @assert ysw ≈ ys
    yspw = smooth(PreprocessedWeightedGraph(g), .1, y)
    @assert yspw ≈ ys

    #Test propagation
    g = grid([5, 5])
    rf = random_forest(g, .5)
    @assert rf.root == rf * collect(1:nv(g))
    p = Partition(rf)
    @assert rf.root ≈ vec(p*rf.root)

end
