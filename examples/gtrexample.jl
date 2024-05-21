using Plots,KirchoffForests,Graphs,LinearAlgebra,Random
pyplot()
rng = MersenneTwister(12345); # Set random seed
g = Graphs.grid([4,4])
p = 8; n = nv(g); y= rand(rng, p); labelednodes = randperm(rng,n)[1:p]; M=I(n)[:,labelednodes]; # Generate an incomplete signal
yprime = M*y
Q = rand(rng, n) # Set regularization parameters
rf = random_forest(g,Q,rng)
xtilde = rf*yprime

p = Partition(rf)
xbar = (p*(yprime .* Q)) ./ (p*Q)
Qdiag = diagm(0=>Q)

L = laplacian_matrix(g);
α = 2*minimum(Q)/(minimum(Q)+maximum(degree(g))) # Chosen as suggested
zbar = xbar - α*((L * xbar) ./ Q + xbar  - yprime)

xexact = inv(L+Qdiag)*Qdiag*yprime

p = Iterators.product(0.0:0.1:0.3, 0.0:0.1:0.3);
xloc = zeros(nv(g))
yloc = zeros(nv(g))
global i = 0
for (x,y) in p
  global i += 1
  xloc[i] = x
  yloc[i] = y  
end

gplotobj = RFGraphPlot(g,xloc,yloc,yprime,15,3,10,:viridis,true,"",15,15,"")
plot(gplotobj)
savefig("gtr-graph.svg");

rfplotobj = RFGraphPlot(SimpleDiGraph(rf),xloc,yloc,[i in rf.roots for i = 1:nv(g)],15,3,1.2,:viridis,false,"",15,15,"")
plot(rfplotobj);
savefig("gtr-forest.svg");


rfplotobj = RFGraphPlot(SimpleDiGraph(rf),xloc,yloc,xtilde,15,3,1.2,:viridis,true,"",15,15,"")
plot(rfplotobj);
savefig("gtr-xtilde.svg");

rfplotobj = RFGraphPlot(SimpleDiGraph(rf),xloc,yloc,xbar,15,3,1.2,:viridis,true,"",15,15,"")
plot(rfplotobj);
savefig("gtr-xbar.svg");

rfplotobj = RFGraphPlot(SimpleDiGraph(rf),xloc,yloc,zbar,15,3,1.2,:viridis,true,"",15,15,"")
plot(rfplotobj);
savefig("gtr-zbar.svg");


rfplotobj = RFGraphPlot(SimpleDiGraph(rf),xloc,yloc,xexact,15,3,1.2,:viridis,true,"",15,15,"")
plot(gplotobj);
savefig("gtr-exact.svg");
