module KirchhoffForests

using LinearAlgebra, SparseArrays, DataStructures
import Base:
    show,
    sum
import Statistics: mean
import StatsBase:
    counts,
    denserank

using Graphs
import Graphs:
    degree,
    inneighbors,
    is_directed,
    nv,
    ne,
    outneighbors,
    SimpleDiGraph

using SimpleWeightedGraphs
import SimpleWeightedGraphs:
    AbstractSimpleWeightedGraph

include("./alias.jl")

export
    random_forest,
    smooth,
    smooth_rf,
    smooth_rf_adapt,
    KirchhoffForest,
    SimpleDiGraph,
    nroots,
    next,
    Partition,
    PreprocessedWeightedGraph,
    reduced_graph,
    smooth_ms,
    self_roots,
    random_successor,
    random_spanning_tree,
    coupled_forests


#=
TODO:
- need clean up using and import
- Too many functions/methods are exposed, with potential overlaps
Suggestion: use a Python inspired approach "import nnumpy as np"
import Graphs; const LG = Graphs
to then call LG.whatever_function
=#


include("forests.jl")
include("random_spanning_tree.jl")
include("smoothing.jl")
include("moments.jl")
include("multiscale.jl")
include("coupled_forests.jl")

end # module
