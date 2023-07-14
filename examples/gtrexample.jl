using RandomForests,Graphs,LinearAlgebra,Random,Plots
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

gplotobj = RFGraphPlot(g,xloc,yloc,yprime,25,3,:viridis,true,"\$\\mathbf{y}'\$","")
plot(gplotobj);
savefig("gtr-graph.svg");

rfplotobj = RFGraphPlot(SimpleDiGraph(rf),xloc,yloc,[i in rf.roots for i = 1:nv(g)],25,3,:viridis,true,"","")
plot(rfplotobj);
savefig("gtr-forest.svg");


rfplotobj = RFGraphPlot(SimpleDiGraph(rf),xloc,yloc,xtilde,25,3,:viridis,true,"\$\\tilde{\\mathbf{x}}\$","")
plot(rfplotobj);
savefig("gtr-xtilde.svg");

rfplotobj = RFGraphPlot(SimpleDiGraph(rf),xloc,yloc,xbar,25,3,:viridis,true,"\$\\bar{\\mathbf{x}}\$","")
plot(rfplotobj);
savefig("gtr-xbar.svg");

rfplotobj = RFGraphPlot(SimpleDiGraph(rf),xloc,yloc,zbar,25,3,:viridis,true,"\$\\bar{\\mathbf{z}}\$","")
plot(rfplotobj);
savefig("gtr-zbar.svg");


gplotobj = RFGraphPlot(g,xloc,yloc,xexact,25,3,:viridis,true,"\$\\hat{\\mathbf{x}}\$","")
plot(gplotobj);
savefig("gtr-exact.svg");
