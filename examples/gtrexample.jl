using RandomForests,Graphs,LinearAlgebra,Random,PyPlot
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

xloc,yloc = grid_layout(4,4);
param = PlotParam(xloc,yloc,false,[],500,3,:viridis,false,"","");
param.signal = yprime;
param.colorbar=true;
param.colorbarlabel=latexstring("{\\mathbf{y}}'");

figure();
plot_graph(g,param=param);
savefig("gtr-graph.svg");

param.showRoots= true;
param.cmap=:viridis;
param.colorbar=true;
param.colorbarlabel="";

figure();
plot_forest(rf,param=param);
savefig("gtr-forest.svg");

param.showRoots= false;
param.cmap=:viridis;
param.signal = xtilde;
param.colorbar=true;
param.colorbarlabel=latexstring("\\tilde{\\mathbf{x}}");
figure();
plot_forest(rf,param=param);
savefig("gtr-xtilde.svg");

param.showRoots= false;
param.cmap=:viridis;
param.signal = xbar;
param.colorbar=true;
param.colorbarlabel=latexstring("\\bar{\\mathbf{x}}");
figure();
plot_forest(rf,param=param);
savefig("gtr-xbar.svg");

param.showRoots= false;
param.cmap=:viridis;
param.signal = zbar;
param.colorbar=true;
param.colorbarlabel=latexstring("\\bar{\\mathbf{z}}");
figure();
plot_forest(rf,param=param);
savefig("gtr-zbar.svg");


param.showRoots= false;
param.cmap=:viridis;
param.signal = xexact;
param.colorbar=true;
param.colorbarlabel=latexstring("\\hat{\\mathbf{x}}");
figure();
plot_graph(g,param=param);
savefig("gtr-exact.svg");
