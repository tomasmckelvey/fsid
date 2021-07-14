using LinearAlgebra, FFTW
using Test

using fsid: 
  fresp,
  estimate_cd,
  estimate_bd,
  transpose_ffdata,
  fresp,
  ffdata2fddata,
  fdestim_bd,
  fdestim_cd,
    fdsid,
    gfdsid,
  ffsid,
  bilinear_c2d,
  bilinear_d2c,
  lsim,
  fdsim,
  cf2df,
  df2cf,
  ltifr
  
# Helper functions
buildss(n, m, p) = randn(n, n), randn(n, m), randn(p, n), randn(p, m)
buildfset(N) = (0:N-1)/N
ztrans(fset) = exp.(2pi * im * fset)
computerr(fd, fde) = norm(fd-fde)/norm(fd)

function timing_fdsid()
    Nset = [100,200,400,800]
    nset = [2,5,10,20]
    m = 2
    p = 8
    MC = 100
    tim_res = zeros(length(Nset),length(nset));
    for Nidx in 1:length(Nset)
        for nidx in 1:length(nset)
            for mc in 1:MC
                N = Nset[Nidx]
                fset = buildfset(N)
                n = nset[nidx]
                A, B, C, D = buildss(n, m, p)
                lam, = eigen(A)
                rho = maximum(abs.(lam)) 
    
                ## Here we create a random stable DT system
                A = A/rho/1.01
                w = 2pi * fset
                z = ztrans(fset)
                fd = fresp(z, A, B, C, D)

                u = randn(N, m)
                y, x = lsim((A, B, C, D), u; type = Real)

                yf = fft(y, 1)
                uf = fft(u, 1)
                fddata = (z, yf, uf)
                tim_res[Nidx,nidx] = tim_res[Nidx,nidx] + @elapsed Ae, Be, Ce, De, xt, s = gfdsid(fddata, n, 2n; estTrans = true, type = Real)
                #fde = fresp(z, Ae, Be, Ce, De)
                #print( computerr(fd, fde))
            end
        end
    end
    return tim_res/MC 
end


# Main test function

print(timing_fdsid())

