#Test the up/down Markov chain alg. for generating USTs

gg = grid([3,3])
# What follows is a probabilistic test that checks that the algorithm samples from the correct distribution
# We use a result by Pemantle & Wilson that states that
# Uniform Spanning Trees are actually a Determinantal Point Process over the edges of the graph
# The kernel of the DPP can be computed explictly from the edge-incidence matrix
IM = Matrix(incidence_matrix(gg;oriented=true));
eg = eigen(IM'*IM);
valid = abs.(eg.values) .> 1e-12
U = eg.vectors[:,valid];
K = U*U';
# The kernel of the DPP gives the inclusion probabilities: for instance K_ii is the probability that edge i is included in a UST
# We compare theoretical to observed incl. probabilities 
AA = [adjacency_matrix(SimpleGraph(random_tree_mc(gg))) for i in 1:100000];
prob_incl = reduce(+,AA)./length(AA);
pr = map((e) -> prob_incl[src(e),dst(e)],edges(gg));
#Check that observed probabilities don't deviate too much 
@test sum(abs.(diag(K)-pr)./pr)./length(pr) < 1e-2
