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


