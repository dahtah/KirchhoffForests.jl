

#@testset "basic" begin
import Random.seed!
seed!(1)
gg = grid([4,4])
for i in 1:100
    check_correctness(random_forest(gg,i/10))
end
# for g in testgraphs(gg)
#     rf = @inferred(random_forest(g))
#     check_correctness(rt.tree,rt.roots)
# end                             

#Try some small graphs
gs = [cycle_graph(5), wheel_graph(9),
      smallgraph(:bull), smallgraph(:tutte)]
map((g) -> (rf=random_forest(g,rand());check_correctness(rf)),gs)
# end
