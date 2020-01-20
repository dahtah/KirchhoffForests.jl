# Moment estimators

#self-roots
function self_roots(rfs :: Vector{RandomForest})
    k = length(rfs)
    n = nv(rfs[1])
    v = 1:n
    for i in 2:k
        v = rfs[i].root[v]
    end
    sum(v .== rfs[1].root)
end

#brute-force spectral estimation
function spectral_est(g :: AbstractGraph,qv,mv,nbins)
#    m = length(qv)
    α = 2*maximum(degree(g))
    grid = LinRange(-5,log.(α),nbins+1)
    #f = (a,b,q) -> (q/(b-a))*(log(q+b) - log(q+a))
    #grid centers
    x = [.5*(grid[i+1]+grid[i]) for i in 1:nbins]
    #form system matrix
    gf = (x,ν) -> 1/(1+exp(x-ν))
    M = reduce(vcat,[ gf.(x,ν) .* (1 .- gf.(x,ν)) for ν in log.(qv)]')
    #M = [ f(grid[i],grid[i+1],q) for q in qv,i in 1:nbins]
    @show size(M)
    w=nonneg_lsq(M,mv,alg=:nnls)
    res = M*w - mv
    @show res
    (x,w,M)
end
