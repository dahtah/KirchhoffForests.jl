# Moment estimators

"""
TODO: doctstring neeeded
"""
#self-roots
function self_roots(rfs::Vector{KirchhoffForest})
    v = rfs[1].root[1:nv(rfs[1])]
    for i in 2:length(rfs)
        v = rfs[i].root[v]
    end
    #return count(v .== rfs[1].root)
    return count(v .== 1:nv(rfs[1]))
end
