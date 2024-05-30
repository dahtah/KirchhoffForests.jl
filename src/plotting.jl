"""
    param = PlotParam
        (xloc :: Array{Number} 
        yloc :: Array{Number}
        showRoots :: Bool
        signal :: Array{Number}
        nodeSize :: Number
        edgeWidth :: Number
        cmap :: Symbol
        colorbar :: Bool
        colorbarlabel :: String
        title :: String)
A struct for configuring plotting parameters of graphs/forests and graph signals. 

# Parameters 
- ```xloc```: Coordinates of nodes in x-axis
- ```yloc```: Coordinates of nodes in y-axis
- ```showRoots``` : Boolean for showing roots of given forests/trees 
- ```signal``` : Signal vector over the vertices
- ```nodeSize``` : Size parameter for nodes 
- ```edgeWidth``` : Width parameter for edges
- ```cmap``` : Color map for colorbar 
- ```colorbar``` : Boolean for showing the color map
- ```colorbarlabel``` : Label for colorbar 
- ```title``` : Title for the figure
"""
mutable struct PlotParam
    xloc :: Array{Number}
    yloc :: Array{Number}
    showRoots :: Bool
    signal :: Array{Number}
    nodeSize :: Number
    edgeWidth :: Number
    cmap :: Symbol
    colorbar :: Bool
    colorbarlabel :: String
    title :: String
end

"""
    plot_graph(g::AbstractGraph;param::PlotParam=PlotParam([],[],false,[],200,2,:viridis,false,"",""))

Function to plot a graph with the signal over the vertices. Plotting parameters can be set by using the struct ```PlotParam```.
"""
function plot_graph(g::AbstractGraph;param::PlotParam=PlotParam([],[],false,[],200,2,:viridis,false,"",""))
    xloc = zeros(nv(g))
    yloc = zeros(nv(g))
    if(!isempty(param.xloc) && !isempty(param.yloc))
        xloc , yloc = param.xloc , param.yloc
    else
        display("Enter x-y positions")
        return 
    end
    if(!isempty(param.signal))
        c = param.signal
        edgecolor = "none"
    else
        c = "white"
        edgecolor = "black"
    end

    for e in edges(g)
        i = e.src
        j = e.dst
        plot([xloc[i],xloc[j]],[yloc[i],yloc[j]],color="black",linewidth=param.edgeWidth)
        if(is_directed(g))
            quiver( xloc[i], yloc[i], (xloc[j]-xloc[i]),(yloc[j]-yloc[i]),headlength=5,headwidth=5, angles="xy", scale_units="xy", scale=1.1)
        end
    end
    axis("off")
    scatter(xloc,yloc,s=param.nodeSize,edgecolors=edgecolor,c=c,zorder=10,cmap=param.cmap)
    if(param.colorbar)
        c= colorbar()
        c.set_label(label=param.colorbarlabel,size=15)
    end
    title(param.title)
end

"""
    plot_tree(rt::NamedTuple;param::PlotParam=PlotParam([],[],true,[],200,2,:viridis,false,"",""))

Function to plot a tree by showing the root or a graph signal over the vertices. Plotting parameters can be set by using the struct ```PlotParam```.
"""
function plot_tree(rt::NamedTuple;param::PlotParam=PlotParam([],[],true,[],200,2,:viridis,false,"",""))
    g = rt.tree
    xloc = zeros(nv(g))
    yloc = zeros(nv(g))
    if(!isempty(param.xloc) && !isempty(param.yloc))
        xloc , yloc = param.xloc , param.yloc
    else
        display("Enter x-y positions")
        return 
    end
    if(!isempty(param.signal) || param.showRoots)
        if(param.showRoots)
            c = zeros(nv(g))
            c[rt.root] = 1.0
            edgecolor = "black"
        else
            c = param.signal
            edgecolor = "none"
        end
    else
        c = "white"
        edgecolor = "black"
    end


    for e in edges(g)
        i = e.src
        j = e.dst
        quiver( xloc[i], yloc[i], (xloc[j]-xloc[i]),(yloc[j]-yloc[i]),headlength=5,headwidth=5, angles="xy", scale_units="xy", scale=1.1)

    end
    axis("off")
    scatter(xloc,yloc,s=param.nodeSize,edgecolors=edgecolor,c=c,zorder=10,cmap=param.cmap)
    if(param.colorbar)
        c= colorbar()
        c.set_label(label=param.colorbarlabel,size=15)
    end
    title(param.title)
end

"""
    plot_forest(rf::KirchoffForest;param::PlotParam=PlotParam([],[],true,[],200,2,:viridis,false,"",""))

Function to plot a forest by showing the roots or a graph signal over the vertices. Plotting parameters can be set by using the struct ```PlotParam```.
"""
function plot_forest(rf::KirchoffForest;param::PlotParam=PlotParam([],[],true,[],200,2,:viridis,false,"",""))
    g = SimpleDiGraph(rf)
    xloc = zeros(nv(g))
    yloc = zeros(nv(g))
    if(!isempty(param.xloc) && !isempty(param.yloc))
        xloc , yloc = param.xloc , param.yloc
    else
        display("Enter x-y positions")
        return 
    end
    if(!isempty(param.signal) || param.showRoots)
        if(param.showRoots)
            c = zeros(nv(g))
            c[collect(rf.roots)] .= 1.0
            edgecolor = "black"
        else
            c = param.signal
            edgecolor = "none"
        end
    else
        c = "white"
        edgecolor = "black"
    end

    for e in edges(g)
        i = e.src
        j = e.dst
        quiver( xloc[i], yloc[i], (xloc[j]-xloc[i]),(yloc[j]-yloc[i]),headlength=5,headwidth=5, angles="xy", scale_units="xy", scale=1.1)
    end
    axis("off")
    scatter(xloc,yloc,s=param.nodeSize,edgecolors=edgecolor,c=c,zorder=10,cmap=param.cmap)
    if(param.colorbar)
        c= colorbar()
        c.set_label(label=param.colorbarlabel,size=15)
    end

    title(param.title)
end

