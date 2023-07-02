push!(LOAD_PATH,"../src/")
using Documenter, RandomForests
using DocumenterCitations
bib = CitationBibliography("./src/references.bib")
makedocs(bib,pages=[
"index.md",
"What's an RSF?"=>"rsf.md",
"RSF-based Graph Tikhonov Regularization" =>"gtr.md",
"Trace Estimation" =>"trace.md",
"Types and Functions" => "typesandfunc.md",
"References" => "references.md"
],sitename="RandomForests.jl")
