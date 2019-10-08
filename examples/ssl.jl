using DelimitedFiles,LightGraphs
using Statistics

function load_citeseer()
    tst = readdlm("/localdata/barthesi/tmp/citeseer/citeseer.content",'\t')
    paper_id = string.(tst[:,1])
    n = length(paper_id)
    class = strip.(string.(tst[:,end]))
    classes = unique(class)
    D = Dict{String,Int64}([(paper_id[i],i) for i in 1:n])
    C = Dict{String,Int64}([(classes[i],i) for i in 1:length(classes)])
    net = strip.(string.(readdlm("/localdata/barthesi/tmp/citeseer/citeseer.cites",'\t')))
    G = SimpleGraph(n)
    m = size(net,1)
    for i in 1:m
        if (haskey(D,net[i,1]) && haskey(D,net[i,2]))
            add_edge!(G,D[net[i,1]],D[net[i,2]])
        end
    end
    class_id = map((v) -> C[v],class)
    Y_true = Matrix(sparse(1:n,class_id,ones(n)))
    (G=G,D=D,C=C,Y_true=Y_true,class_id=class_id)
end

function test_ssl_citeseer(G,C,Y_true;frac=.2,q=.2,method="exact")
    n = size(Y_true,1)
    train = (rand(n) .< frac)
    class_id = map((v) -> v[2],argmax(Y_true,dims=2))
    Y = copy(Y_true)

    Y[.!train,:] .= 0
    est_cl = ssl(G,Y,q,method=method)
    mean(est_cl .== class_id)
end

#label propagation
function ssl(G,Y,q;method="exact",nrep=10)
    has_label = sum(Y,dims=1) .!= 0
    if (method=="exact")
        S = smooth(G,q,Y)
    elseif (method=="xtilde")
        S = smooth_rf(G,q,Y;nrep=nrep,variant=1).est
    elseif (method=="xbar")
        S = smooth_rf(G,q,Y;nrep=nrep,variant=2).est
    end
    map((v) -> v[2],argmax(S,dims=2))
end
