@testset "smoothing" begin
    g=grid([11],periodic=true)
    #t = LinRange(0,1,10)
    t = (0:10)./11
    y = cos.(2*pi*t)
    ys = smooth(g,.1,y)
    @test (ys/maximum(ys)) ≈ y #should be eigenvector
    ysw = smooth(SimpleWeightedGraph(g),.1,y)
    @test ysw ≈ ys
    yspw = smooth(PreprocessedWeightedGraph(g),.1,y)
    @test yspw ≈ ys

    #Test propagation
    g = grid([5, 5])
    rf = random_forest(g,.5)
    @assert rf.root == rf*collect(1:nv(g))
    p = Partition(rf)
    @test rf.root ≈ vec(p*rf.root)

end
