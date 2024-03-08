using Images, ImageMagick, ImageView, Gtk.ShortNames, TestImages
#=
TODO: describe the example
=#

function denoise(im; q=.4, nrep=20)
    nx, ny = size(im)
    G = grid([nx, ny])

    y = im[:]
    xhat = smooth(G, q, y)
    xtilde = smooth_rf(PreprocessedWeightedGraph(G), q, y; variant=1, nrep=nrep).est
    xbar = smooth_rf(PreprocessedWeightedGraph(G), q, y; variant=2, nrep=nrep).est

    gr, frames, canvases = canvasgrid((2, 2))  # 1 row, 2 columns
    rs = (v) -> reshape(v, nx, ny)
    imshow(canvases[1, 1], rs(y))
    imshow(canvases[1, 2], rs(xhat))
    imshow(canvases[2, 1], rs(xtilde))
    imshow(canvases[2, 2], rs(xbar))
    win = Window(gr)
    Gtk.showall(win)  # TODO not sure GTK is in scope
end

#run the denoising demo on an image from the testimage library
#Try:
#demo_denoise("lena")
#demo_denoise("fabio_gray_512")
function demo_denoise(imname="lena", q=.6)
    im = Float64.(Gray.(testimage(imname))) + randn(size(im))
    denoise(y; q=q);
end
