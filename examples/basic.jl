using Graphs,RandomForests
#random graph
G = SimpleGraph(5,10)
#random signal
y = randn(5)
q = .1
x̂ = smooth(G,q,y)
x̃ = smooth_rf(G,q,y).est
x̄ = smooth_rf(G,q,y;variant=2).est

e1 = sqrt( sum((x̂ .- x̃).^2))
e2 = sqrt( sum((x̂ .- x̄).^2))
#should happen with high prob.
@assert e2 > e1
