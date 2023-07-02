@testset "cv" begin
    Random.seed!(1)
    n = 1000
    gg = barabasi_albert(n,10)
    L = laplacian_matrix(gg)
    for i in 1:100
        rf = random_forest(gg,i)
        bt = partition_boundary_track(gg,rf)
        p = Partition(rf)
        part=Int.(denserank(rf.root))
        sizep = counts(part)
        diag_S = 1 ./ sizep[part]

        Sbar = p*Matrix(I(n))
        @test tr(L*Sbar) ≈ (bt'*diag_S)

        bt = root_boundary_track(gg,rf)
        Stilde = rf*Matrix(I(n))
        @test tr(L*Stilde) ≈ sum(bt)
    end
end
