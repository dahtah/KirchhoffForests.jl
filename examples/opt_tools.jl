using LightGraphs
using RandomForests
using LinearAlgebra
using SparseArrays
using Plots
using StatsBase
using Clustering
using Distributions
using ImageView, TestImages, Gtk.ShortNames, Images, ImageMagick, ImageQualityIndexes

plotly()
#
# function newton(G,y,t0,mu;α=0.001,numofiter = 100,tol=0.001, method="exact",nrep=100)
#     t_k = copy(t0)
#     tprev = copy(t0)
#     increment = norm(t0)
#     L = laplacian_matrix(G)
#     k = 0
#
#     while( increment > tol && k < numofiter)
#         vec_k = 1.0 .- (y ./ exp.(t_k)) + ((L*t_k)./(mu .* exp.(t_k)))
#         tprev = copy(t_k)
#         println("Iteration $k, increment $increment")
#
#         if (method=="exact")
#             t_k -= α*(smooth(G,mu .* exp.(t_k),vec_k))
#         else
#             vecbar = repeat(mean(vec_k, dims=1), outer=size(y,1))
#             Δ = -vecbar + vec_k
#             if (method=="xtilde")
#                 t_k -= α*(vecbar + smooth_rf(G,1 ./ (mu .* exp.(t_k)),Δ,[];nrep=nrep,variant=1).est)
#             elseif (method=="xbar")
#                 t_k -= α*(vecbar + smooth_rf(G,1 ./ (mu .* exp.(t_k)),Δ,[];nrep=nrep,variant=2).est)
#             end
#         end
#         k += 1
#         increment = norm(tprev - t_k)
#     end
#     println("Method: $method. Terminated after $k iterations, increment $increment")
#     return exp.(t_k)
# end
#
# function irls(G,y,z0,mu;numofiter = 100,tol=0.001, method="exact",nrep=100)
#     B = incidence_matrix(G,oriented=true)
#     k = 0
#     increment = norm(z0)
#     z_k = copy(z0)
#     z_prev = copy(z0)
#     while( increment > tol && k < numofiter)
#         zprev = copy(z_k)
#         M_k = spdiagm(0 => (B'*z_k).^(-1))
#         L_k = B*M_k*(B')
#         L_k[diagind(L_k)] .= 0
#         G = SimpleGraph(-(L_k))
#         println("Iteration $k, increment $increment")
#
#         if (method=="exact")
#             z_k = (smooth(G,mu,y))
#         else
#             ybar = repeat(mean(y, dims=1), outer=size(y,1))
#             Δ = -ybar + y
#             if (method=="xtilde")
#                 z_k = (ybar + smooth_rf(G,mu,Δ,[];nrep=nrep,variant=1).est)
#             elseif (method=="xbar")
#                 z_k = (ybar + smooth_rf(G,mu,Δ,[];nrep=nrep,variant=2).est)
#             end
#         end
#         k += 1
#         increment = norm(zprev - z_k)
#     end
#     println("Method: $method. Terminated after $k iterations, increment $increment")
#
#     return z_k
# end


imname = "lake_gray"
im = imresize(testimage(imname), 64, 64)
im = Int64.(floor.(Float64.(Gray24.(im))*256))
nx = size(im,1)
ny = size(im,2)
rs = (v) -> reshape(v,nx,ny)
G = LightGraphs.grid([nx,ny])
im = im[:]
im_noisy = zeros(size(im))
for (idx,i) in enumerate(im)
    im_noisy[idx] = rand(Poisson(i))
end
# imshow(reshape(im,nx,ny))
x = zeros(size(im))
xtilde = zeros(size(im))
xbar = zeros(size(im))
# for i = 1 : k
y = im_noisy
z0 = exp.(rand(nv(G)))

x = Int64.(floor.(newton(G,y,z0,1.0;numofiter = 100,tol=0.001, method="exact")))
xtilde = Int64.(floor.(newton(G,y,z0,1.0;numofiter = 100,tol=0.001, method="xtilde",nrep=10)))
xbar = Int64.(floor.(newton(G,y,z0,1.0;numofiter = 100,tol=0.001, method="xbar",nrep=10)))
# end
Gray.(reshape(y./256,nx,ny))
Gray.(reshape(x./256,nx,ny))
Gray.(reshape(xtilde./256,nx,ny))
Gray.(reshape(xbar./256,nx,ny))

# imshow(reshape(x,nx,ny))
# imshow(reshape(xtilde,nx,ny))
# imshow(reshape(xbar,nx,ny))
# display(ImageQualityIndexes.assess_psnr(x, im))
# display(ImageQualityIndexes.assess_psnr(xtilde, im))
# display(ImageQualityIndexes.assess_psnr(xbar, im))
