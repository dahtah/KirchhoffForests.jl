using RandomForests,LightGraphs,StatsBase
using Convex,SCS,GLPK,GRUtils,JuMP


function approx_moments(g,qmin,qmax,tol=.001,nrep=10)
    ed = laplacian_matrix(g) |> Matrix |> eigen
    n = nv(g)
    ev = ed.values[2:end]
    lqv = LinRange(log(qmin),log(qmax),30)
    qv = exp.(lqv)
    pr = (q) -> @. q/(q+ev)
    x = log.(ev)
    ms = (q) -> sum(pr(q).*(1 .- pr(q)))
    #    ms_est = (q) -> self_roots([random_forest(g,q) for _ in 1:2])
    ms_est =(q) -> est_moment(g,q,nrep)
    mest = ms_est.(qv)
    mest[mest .< 0] .= 0
    #mest = ms.(qv)
    log_grid,w,M = spectral_est(g,qv,mest,201);
    @show size(M)
    tol = tol*nv(g)
    ii,lbound=est_bound(M,mest,nv(g),tol,:lower)
    ii,ubound=est_bound(M,mest,nv(g),tol,:upper)
    gr = exp.(log_grid);
    subplot(1,2,1)
    plot(lqv,ms.(qv)/n,"-b",lqv,(M*w)/n,"-r",lqv,(mest)/n,"--r")
#    oplot(lqv,(v)->exp(v)/((1+exp(v))^2))
    subplot(1,2,2)
    plot(log_grid,cumsum(w[:])/sum(w[:]),"-r",
    log_grid,ecdf(ev).(gr),"-b",
    log_grid[ii],ubound/nv(g),"--k",log_grid[ii],lbound/nv(g),"--k")
    gcf()
end

function solve_lp_jump(M,y,v,γ,nv)
    m,n = size(M)
    model = Model(Clp.Optimizer)
    @variable(model,0 <= w[1:n]);
    @objective(model,Min,dot(v,w));
    @constraint(model,(M*w .- y) .<= γ);
    @constraint(model,-γ .<= (M*w .- y));
    @constraint(model,sum(w) == nv)
    optimize!(model)
    objective_value(model)
end

function est_bound(M,y,nv,γ=.5,type=:lower)
    m,n = size(M)
    w = Variable(n)
    #    solver = () -> SCS.Optimizer(verbose=0)
    solver = () -> GLPK.Optimizer()
    #solver = () -> Clp.Optimizer(verbose=0)
    ii = 1:10:n
    v = zeros(n)
    p = minimize(dot(w,v))
    @show size(v)
    #ws = [];
    p.constraints += abs(M*w - y) <= γ
    p.constraints += w >= 0
    p.constraints += sum(w) == nv
    bound = zeros(length(ii))
    for i in 1:length(ii)
        v[1:ii[i]] .= (type == :lower ? -1 : 1)
        #solve!(p,solver,warmstart=true)
        solve!(p,solver)
        #push!(ws,w.value)
        #val_check = solve_lp_jump(M,y,v,γ,nv)
        bound[i] = (type == :lower ? -p.optval : p.optval)
        #@show val_check,p.optval
    end
    (ii,bound)
end

function projection_bounds(g,partition)
    gr = reduced_graph(g,partition)
    ed = laplacian_matrix(g) |> Matrix |> eigen
    edr = laplacian_matrix(gr) |> Matrix |> eigen
    γ₁,γ₂ = extrema(partition.sizep)
    @show γ₁,γ₂
    #Lred = lapla
    # P = sparse(1:nv(g),partition.part,ones(nv(g)))
    # Ps = P ./ sqrt.(sum(P.^2,dims=1))
    # Lred = Ps'*laplacian_matrix(g)*Ps
    # Lred2 = RandomForests.laplacian_matrix(gr) |> Matrix
    # w = Vector(partition.sizep)
    # Lred2 = diagm(1 ./ sqrt.(w))*Lred2*diagm( 1 ./ sqrt.(w))

    #
    #edr = diagm(1 ./ sqrt.(w))*Lred*diagm( 1 ./ sqrt.(w))|> eigen
    #edr = eigen(Matrix(Lred))
    # Lred = laplacian_matrix(g)[1:Int(nv(g)/2),1:Int(nv(g)/2)]
    # edr = eigen(Matrix(Lred))
    x = LinRange(minimum(ed.values),maximum(ed.values),100)
    ii = collect(1:length(edr.values))
    plot((1:nv(g))/nv(g),ed.values,"-b",
         ii/nv(g),edr.values/γ₁,"--r",
         (nv(g) - nv(gr) .+ ii)/nv(g),edr.values/γ₂,"--r"
         )

    # plot(x,ecdf(ed.values).(x),"-b",
    #      edr.values,ii/nv(g),"--r",
    #      edr.values,(nv(g) - nv(gr) .+ ii)/nv(g),"--r"
    #      )
    #plot(edr.values,(nv(g) - nv(gr) .+ ii)/nv(g),"--r")
#    (edr.values,(nv(g) - nv(gr) .+ ii)/nv(g))
end

# function est_moment(g,q)
#     rf = [random_forest(g,q) for _ in 1:2]
#     nr = [r.nroots for r in rf]
#     max(mean(nr) - self_roots(rf) - 1,0) 
# end

#use a series expansion when q is large enough
function est_moment(g,q,nrep=2)
    α = 2*maximum(degree(g))
    if (q > 40*α)
        sum(degree(g))/q 
    else
        rf = [random_forest(g,q) for _ in 1:nrep]
        nr = [r.nroots for r in rf]
        #max(mean(nr) - self_roots(rf),0)
        #@show self_roots(rf)
        mean(nr) - self_roots(rf,2)
    end
end

function self_roots_naive(rfs,k)
    l = length(rfs)
    res = Vector{Int}()
    i = 1
    while i+k-1 <= l
        push!(res,self_roots(rfs[i:(i+k-1)]))
        i = i+k
    end
    mean(res)
end
#Demo:
#g = LightGraphs.grid([25,25])
#approx_moments(g,1e-3,1e5,.1)
