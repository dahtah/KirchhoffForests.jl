using RandomForests,LightGraphs,StatsBase,LinearAlgebra
using Convex,SCS,GLPK,GRUtils,JuMP, RCall,Clp


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

function approx_moments_ho(g,w,qmin,qmax,tol=.001,nrep=10;slack=:absolute,exact=:false)
    ed = laplacian_matrix(g) |> Matrix |> eigen
    n = nv(g)
    ev = ed.values[2:end]
    lqv = LinRange(log(qmin),log(qmax),50)
    qv = exp.(lqv)
    pr = (q) -> @. q/(q+ev)
    x = log.(ev)
    ms = (q) -> sum([w[i]*sum(pr(q).^i) for i in 1:length(w)])
#    ss = (q) -> sum((pr(q) .* (1 .- pr(q))).^2)
    ms_est =(q) -> est_linear_moment(g,q,w,nrep)
    mest = (exact ? ms.(qv) : ms_est.(qv) )
    #mest = ms.(qv)
    mest[mest .< 0] .= 0
    log_grid,M = RandomForests.deconv_matrix(g,w,qv,201);
    est_nn = RandomForests.solve_nneg(M,mest)
    if (slack == :absolute)
        tol = tol*nv(g)
    elseif (slack == :relative)
        zz = [est_linear_moment(g,1.0,w) for _ in 1:10]
        tol = tol*std(zz)
        
    end
    rel_err = mest ./ ms.(qv)
    @show extrema(rel_err),tol
#    plot(log.(qv),sqrt.(mest),"-r",log.(qv),sqrt.(ms.(qv)),"--b")
    ii,lbound=est_bound(M,mest,nv(g),tol,:lower;slack=slack)
    ii,ubound=est_bound(M,mest,nv(g),tol,:upper;slack=slack)
    gr = exp.(log_grid);
    subplot(1,2,1)
    #plot(lqv,ms.(qv))
    plot(lqv,ms.(qv)/n,"-b",lqv,(M*est_nn)/n,"-r",lqv,(mest)/n,"--r")
    #    oplot(lqv,(v)->exp(v)/((1+exp(v))^2))
    subplot(1,2,2)
    plot(log_grid,cumsum(est_nn[:])/sum(est_nn[:]),"-r",
         log_grid,ecdf(ev).(gr),"-b",
         log_grid[ii],ubound/nv(g),"--k",log_grid[ii],lbound/nv(g),"--k")
    gcf()
end

function approx_moments_stacking(g,order,qmin,qmax,tol=.001;slack=:absolute,exact=:false,grid_size=201,nrep=10)
    ed = laplacian_matrix(g) |> Matrix |> eigen
    n = nv(g)
    ev = ed.values[2:end]
    lqv = LinRange(log(qmin),log(qmax),30)
    qv = exp.(lqv)
    pr = (q) -> @. q/(q+ev)
    x = log.(ev)
    ms = (q) -> [sum(pr(q).^i) for i in 1:order]
#    ss = (q) -> sum((pr(q) .* (1 .- pr(q))).^2)
    ms_est =(q) -> est_moments(g,q,order,nrep)
    mf = (exact ? ms : ms_est)
    
    #mest = ms.(qv)
#    mest[mest .< 0] .= 0
    log_grid,M = RandomForests.deconv_matrix(g,qv,grid_size);
    A = reduce(vcat,[M.^i for i in 1:order])
    R"
    A = $A
    imageM(A)
    "
    y = reduce(vcat,[mf(q)' for q in qv])[:]
    R"    y = $y "
    @show size(A),size(y)
    est_nn = RandomForests.solve_nneg(A,y)
    
    if (slack == :absolute)
        tol = tol*nv(g)
    elseif (slack == :relative)
        zz = [est_linear_moment(g,1.0,w) for _ in 1:10]
        tol = tol*std(zz)
    end
    
    #rel_err = mest ./ ms.(qv)
    #@show extrema(rel_err),tol
    #    plot(log.(qv),sqrt.(mest),"-r",log.(qv),sqrt.(ms.(qv)),"--b")
    ii,lbound=est_bound_jump(A,y,n,tol,:lower;slack=slack)
    ii,ubound=est_bound_jump(A,y,n,tol,:upper;slack=slack)
    gr = exp.(log_grid);
    # 
    # #plot(lqv,ms.(qv))
    # plot(lqv,ms.(qv)/n,"-b",lqv,(M*est_nn)/n,"-r",lqv,(mest)/n,"--r")
    # #    oplot(lqv,(v)->exp(v)/((1+exp(v))^2))

    subplot(1,2,1)
    plot(1:length(y),y,"--r",1:length(y),A*est_nn,"*b")
    subplot(1,2,2)
    plot(log_grid,cumsum(est_nn[:])/sum(est_nn[:]),"-r",
         log_grid,ecdf(ev).(gr),"-b",
         log_grid[ii],ubound/(nv(g)-1),"--k",log_grid[ii],lbound/(nv(g)-1),"--k")
    gcf()                       
end


function approx_moments_test(order,qmin,qmax,tol=.001;slack=:absolute,exact=:true,grid_size=201,nrep=10)
    #ev = exp.([-4,-3,-2,-1,0])
    ev = rand(10)
    n = length(ev)
    g = LightGraphs.grid([n])
    lqv = LinRange(log(qmin),log(qmax),50)
    qv = exp.(lqv)
    pr = (q) -> @. q/(q+ev)
    x = log.(ev)
    ms = (q) -> [sum(pr(q).^i) for i in 1:order]
#    ss = (q) -> sum((pr(q) .* (1 .- pr(q))).^2)
    #ms_est =(q) -> est_moments(g,q,order,nrep)
    mf = ms
    
    #mest = ms.(qv)
#    mest[mest .< 0] .= 0
    log_grid,M = RandomForests.deconv_matrix(g,qv,grid_size);
    A = reduce(vcat,[M.^i for i in 1:order])
    R"
    A = $A
    imageM(A)
    "
    y = reduce(vcat,[mf(q)' for q in qv])[:]
    R"    y = $y "
    @show size(A),size(y)
    est_nn = RandomForests.solve_nneg(A,y)
    @show est_nn
    # if (slack == :absolute)
    #     tol = tol*nv(g)
    # elseif (slack == :relative)
    #     zz = [est_linear_moment(g,1.0,w) for _ in 1:10]
    #     tol = tol*std(zz)
    # end
    err = A*est_nn - y
    R"
    err = $err
    plot(err)
    "

    #rel_err = mest ./ ms.(qv)
    #@show extrema(rel_err),tol
    #    plot(log.(qv),sqrt.(mest),"-r",log.(qv),sqrt.(ms.(qv)),"--b")

    ii,lbound=est_bound_jump(A,y,n,tol,:lower;slack=slack)
    ii,ubound=est_bound_jump(A,y,n,tol,:upper;slack=slack)
    gr = exp.(log_grid);
    # 
    # #plot(lqv,ms.(qv))
    # plot(lqv,ms.(qv)/n,"-b",lqv,(M*est_nn)/n,"-r",lqv,(mest)/n,"--r")
    # #    oplot(lqv,(v)->exp(v)/((1+exp(v))^2))

    subplot(1,2,1)
    plot(1:length(y),y,"--r",1:length(y),A*est_nn,"*b")
    subplot(1,2,2)
    plot(log_grid,cumsum(est_nn[:])/sum(est_nn[:]),"-r",
         log_grid,ecdf(ev).(gr),"-b",
         log_grid[ii],ubound/n,"--k",log_grid[ii],lbound/(n),"--k")
    gcf()                       
end




function solve_lp_jump(M,y,v,γ,nv)
    m,n = size(M)
    model = Model(GLPK.Optimizer)
    @variable(model,0 <= w[1:n]);
    @objective(model,Max,dot(v,w));
    @constraint(model,(M*w .- y) .<= γ);
    @constraint(model,-γ .<= (M*w .- y));
    @constraint(model,sum(w) == nv)
    optimize!(model)
    wv = value.(w)
    err = M*wv - y
    @show γ,extrema(err),termination_status(model)
    R"
    plot($y)
    pointsr($M %*% $wv)
    "

    objective_value(model)
end

function est_bound_jump(M,y,nv,γ=.5,type=:lower;slack=:absolute)
    m,n = size(M)
    ii = 1:10:n
    v = zeros(n)
    bound = zeros(length(ii))
    for i in 1:length(ii)
        v[1:ii[i]] .= (type == :lower ? -1 : 1)
        val = solve_lp_jump(M,y,v,γ,nv)
        bound[i] = (type == :lower ? -val : val)
        #@show val_check,p.optval
    end
    (ii,bound)
end


function est_bound(M,y,nv,γ=.5,type=:lower;slack=:absolute)
    m,n = size(M)
    w = Variable(n)
    #    solver = () -> SCS.Optimizer(verbose=0)
    solver = () -> GLPK.Optimizer()
    #solver = () -> Clp.Optimizer()
    ii = 1:10:n
    v = zeros(n)
    p = minimize(dot(w,v))
    @show size(v)
    #ws = [];
    if (slack == :absolute)
        p.constraints += abs(M*w - y) <= γ
    elseif (slack == :relative)
        p.constraints += abs(M*w - y) <= γ.*sqrt.(y)
    end
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

function est_moments(g,q,order,nrep=4)
    @assert nrep >= order
    α = 2*maximum(degree(g))
    # if (q > 60*α)
    #     trL = sum(degree(g))
    #     ms = [(nv(g) - (i/q)*trL) for i in 1:order]
    #     ms .- 1
    # else
        rf = [random_forest(g,q) for _ in 1:nrep]
        nr = [RandomForests.self_roots(rf,i) for i in 1:order]
        nr .- 1
#    end
end

function plot_moment_relvar(g)
    qval = exp.(LinRange(-6,4,10))
    ss = (q,i) -> (zz=([est_moments(g,q,i,i)[i] for _ in 1:300]);(mean(zz),std(zz)))
#    ss = (q,i) -> (zz=([est_moments(g,q,i,5)[i] for _ in 1:300]);(mean(zz),std(zz)))
    rv = (q,i) -> (zz=ss(q,i);zz[2]/zz[1])
    meanv = (q,i) -> (zz=ss(q,i);zz[1])
    stdv = (q,i) -> (zz=ss(q,i);zz[2])

#    relvar = [rv.(qval,i) for i in 1:5]
    m = [meanv.(qval,i) for i in 1:5]
    sd = [stdv.(qval,i) for i in 1:5]
    R"
    df = data.frame(m=do.call(c,$m),sd=do.call(c,$sd),k=rep(1:5,each=length($qval)),q=$qval)
    library(ggplot2)
    ggplot(df,aes(log(q),m,col=as.factor(k)))+geom_line()
    "
    R"
    ggplot(df,aes(log(q),sd,col=as.factor(k)))+geom_line()
    "
    R"
    ggplot(df,aes(log(q),sd/m,col=as.factor(k)))+geom_line()
    "
    plot(log.(qval),relvar[1],"--r",log.(qval),relvar[2],"--g",log.(qval),relvar[3],"--b")
    plot(log.(qval),m[1],"--r",log.(qval),m[2],"--g",log.(qval),m[3],"--b")

end

function est_linear_moment(g,q,w,nrep=4)
    @assert nrep >= length(w)
    α = 2*maximum(degree(g))
    if (q > 60*α)
        trL = sum(degree(g))
        ms = sum([w[i]*(nv(g) - (i/q)*trL) for i in 1:length(w)])

    else
        #    
        rf = [random_forest(g,q) for _ in 1:nrep]
        nr = [RandomForests.self_roots(rf,i) for i in 1:length(w)]
        dot(w,nr)
    end
end


#Demo:
#g = LightGraphs.grid([25,25])
#approx_moments(g,1e-3,1e5,.1)
