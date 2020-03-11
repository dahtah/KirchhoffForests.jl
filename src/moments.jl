# Moment estimators



#self-roots
function self_roots(rfs :: Vector{RandomForest})
    k = length(rfs)
    n = nv(rfs[1])
    v = rfs[1].root
    for i in 2:k
        v = rfs[i].root[v]
    end
    sum(v .== 1:n)
end

function self_roots(rfs :: Vector{RandomForest},order)
    @assert order<=length(rfs)
    l = length(rfs)
    res = Vector{Int}()
    for cc in combinations(1:l,order)
        #looping over permutations is overkill, need to improve this bit
        #some permutations yield identical results, eg. when order==2
        for pp in permutations(1:order) 
            push!(res,self_roots(rfs[cc][pp]))
        end
    end
    mean(res)
end

#brute-force spectral estimation
function deconv_matrix(g :: AbstractGraph,qv,nbins)
#    m = length(qv)
    α = 2*maximum(degree(g))
    grid = LinRange(-5,log.(α),nbins+1)
    #f = (a,b,q) -> (q/(b-a))*(log(q+b) - log(q+a))
    #grid centers
    x = [.5*(grid[i+1]+grid[i]) for i in 1:nbins]
    #form system matrix
    gf = (x,ν) -> 1/(1+exp(x-ν))
    M = reduce(vcat,[ gf.(x,ν)  for ν in log.(qv)]')
    (x,M)
end

function solve_nneg(M,y)
    nonneg_lsq(M,y,alg=:nnls)
end

#brute-force spectral estimation
function spectral_est(g :: AbstractGraph,qv,mv,nbins)
    α = 2*maximum(degree(g))
    grid = LinRange(-5,log.(α),nbins+1)
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

