using KirchoffForests,Graphs,LinearAlgebra,PyPlot, StatsBase,ProgressBars
n = 10000
g = barabasi_albert(n,10)
L = Matrix(laplacian_matrix(g))
q = 1.0
tr_exact = tr(q*inv(L+q*I)) 

α=2*q/(2*q+mean(degree(g)))
NREPRANGE = Int64.(round.(10 .^ (1:0.1:2)))

err_tr_est = zeros(length(NREPRANGE))
err_tr_cv_tilde = zeros(length(NREPRANGE))
err_tr_cv_bar = zeros(length(NREPRANGE))
err_tr_st = zeros(length(NREPRANGE))

EXPREP = 1000
for er = ProgressBar(1:EXPREP)
    for (idx,NREP) in enumerate(NREPRANGE)
        tr_est = trace_estimator(g,q;variant=1,NREP=NREP) # The forest estimator s
        err_tr_est[idx] += abs(tr_est - tr_exact) / tr_exact
        tr_est_cv_tilde = trace_estimator(g,q;variant=2,α=α,NREP=NREP) # s_tilde 
        err_tr_cv_tilde[idx] += abs(tr_est_cv_tilde - tr_exact) / tr_exact
        tr_est_cv_bar = trace_estimator(g,q;variant=3,α=α,NREP=NREP) # s_bar
        err_tr_cv_bar[idx] += abs(tr_est_cv_bar - tr_exact) / tr_exact
        tr_est_st = trace_estimator(g,q;variant=4,α=α,NREP=NREP) # stratified estimator s_st
        err_tr_st[idx] += abs(tr_est_st - tr_exact) / tr_exact
    end
end

plot(NREPRANGE,err_tr_est ./ EXPREP,label="s",marker="o")
plot(NREPRANGE,err_tr_cv_tilde ./ EXPREP,label="\$ \\tilde{s}\$",marker="x")
plot(NREPRANGE,err_tr_cv_bar ./ EXPREP,label="\$ \\bar{s}\$",marker="d")
plot(NREPRANGE,err_tr_st ./ EXPREP,label="\$ s_{st}\$",marker="s")

xlabel("Number of Samples",fontsize=20)
ylabel("Relative Error \$ \\frac{|\\cdot - tr(\\mathsf{K})|}{tr(\\mathsf{K})} \$",fontsize=20)
xticks(fontsize=15)
yticks(fontsize=15)
yscale("log")
xscale("log")
legend(fontsize=15)
tight_layout()

savefig("../docs/src/trace_comp")
