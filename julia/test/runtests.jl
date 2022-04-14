using LinearAlgebra, FFTW
using Test


using fsid:
    fresp,
    estimate_cd,
    estimate_bd,
    transpose_ffdata,
    ffdata2fddata,
    fdestim_bd,
    fdestim_cd,
    fdsid,
    ffsid,
    bilinear_c2d,
    bilinear_d2c,
    lsim,
    fdsim,
    cf2df,
    df2cf,
    ltifr,
    lrm,
    kung_realization,
    markov,
    make_sys_real,
    make_obs_real,
    moebius,
    moebius_inv,
    moebius_arg_inv,
    moebius_arg


# Helper functions
buildss(n, m, p) = randn(n, n), randn(n, m), randn(p, n), randn(p, m)
buildfset(N) = (0:N-1)/N
ztrans(fset) = exp.(2pi * im * fset)
computerr(fd, fde) = norm(fd-fde)/norm(fd)


const TOL = 1e-8


# Unit test functions
function unit_test_estimate_cd_1(fset, m, n, p)
    z = ztrans(fset)
    A, B, C, D = buildss(n, m, p)
    fd = fresp(z, A, B, C, D)
    Ce, De = estimate_cd(fd, z, A, B)
    fde = fresp(z, A, B, Ce, De)

    @test computerr(fd, fde) < TOL
end


function unit_test_estimate_cd_2(fset, m, n, p)
    z = ztrans(fset)
    A, B, C, D = buildss(n, m, p)
    D = zeros(p, m)
    fd = fresp(z, A, B, C, D)
    Ce, De = estimate_cd(fd, z, A, B; estimd = false)
    fde = fresp(z, A, B, Ce, De)

    @test computerr(fd, fde) < TOL
end


function unit_test_estimate_cd_3(fset, m, n, p)
    z = ztrans(fset)
    A, B, C, D = buildss(n, m, p) .+ im .* buildss(n, m, p)
    fd = fresp(z, A, B, C, D)
    Ce, De = estimate_cd(fd, z, A, B; type = Complex)
    fde = fresp(z, A, B, Ce, De)

    @test computerr(fd, fde) < TOL
end


function unit_test_estimate_bd(fset, m, n, p)
    z = ztrans(fset)
    A, B, C, D = buildss(n, m, p)
    fd = fresp(z, A, B, C, D)
    Be, De = estimate_bd(fd, z, A, C)
    fde = fresp(z, A, Be, C, De)

    @test computerr(fd, fde) < TOL
end


function unit_test_estimate_bd_2(fset, m, n, p)
    z = ztrans(fset)
    A, B, C, D = buildss(n, m, p)
    D = zeros(p,m)
    fd = fresp(z, A, B, C, D)
    Be, De = estimate_bd(fd, z, A, C, estimd=false)
    fde = fresp(z, A, Be, C, De)

    @test computerr(fd, fde) < TOL
end


function unit_test_estimate_bd_3(fset, m, n, p)
    z = ztrans(fset)
    A, B, C, D = buildss(n, m, p) .+ im .* buildss(n, m, p)
    fd = fresp(z, A, B, C, D)
    Be, De = estimate_bd(fd, z, A, C, type=Complex)
    fde = fresp(z, A, Be, C, De)

    @test computerr(fd, fde) < TOL
end


function unit_test_transpose_ffdata(fset, m, n, p)
    z = ztrans(fset)
    A, B, C, D = buildss(n, m, p)
    fd = fresp(z, A, B, C, D)
    fde = transpose_ffdata(transpose_ffdata(fd))

    @test computerr(fd, fde) < TOL
end


function unit_test_fresp(fset, m, n, p)
    z = ztrans(fset)
    T = randn(n, n)
    _, s, = svd(T)
    T = s[n] < 1e-6 ? T + 1e-6*I : T
    _, B, C, D = buildss(n, m, p)

    nw = length(z)
    dd = zeros(Complex, n)
    adiag = randn(n)
    A = diagm(0 => adiag)
    fd = Array{Complex}(undef, nw, p, m)

    for fidx in 1:nw
        emad = z[fidx] * ones(n)
        for didx in 1:n
            dd[didx] = 1 / (emad[didx] - adiag[didx])
        end
        fd[fidx, :, :] = ((C * diagm(0 => dd)) * B) + D
    end

    Ae = T * (A/T)
    Be = T * B
    Ce = C/T
    De = D
    fde = fresp(z, Ae, Be, Ce, De) #SLOW?
    fdef = fresp(z, Ae, Be, Ce, De)
    @test computerr(fd, fde) < TOL
    @test computerr(fd, fdef) < TOL
end

function unit_test_fresp_def(fset, m, n, p)
    z = ztrans(fset .+ 1.0e-5)
    A = [0 0; 1 0]
    _, B, C, D = buildss(n, m, p)
    frsp = fresp(z, A, B, C, D)  #SLOW?
    frspf = fresp(z, A, B, C, D)

    @test computerr(frspf, frsp) < TOL
end


function unit_test_fdestim_bd(fset, m, n, p)
    A, B, C, D = buildss(n, m, p)
    w = 2pi * fset
    z = ztrans(fset)
    fd = fresp(z, A, B, C, D)
    U, Y, wn = ffdata2fddata(fd, w)
    zn = exp.(im * wn)
    Be, De, resid = fdestim_bd(zn, Y, U, A, C)
    fde = fresp(z, A, Be, C, De)

    @test computerr(fd, fde) < TOL
end


function unit_test_fdestim_bd_no_d(fset, m, n, p)
    A, B, C, D = buildss(n, m, p)
    D = zeros(p,m)
    w = 2pi * fset
    z = ztrans(fset)
    fd = fresp(z, A, B, C, D)
    U, Y, wn = ffdata2fddata(fd, w)
    zn = exp.(im * wn)
    Be, De, resid = fdestim_bd(zn, Y, U, A, C, estimd=false)
    fde = fresp(z, A, Be, C, De)

    @test computerr(fd, fde) < TOL
end


function unit_test_fdestim_bd_cmplx(fset, m, n, p)
    A, B, C, D = buildss(n, m, p)
    B = B + im*randn(n, m)
    D = D + im*randn(p, m)
    w = 2pi * fset
    z = ztrans(fset)
    fd = fresp(z, A, B, C, D)
    U, Y, wn = ffdata2fddata(fd, w)
    zn = exp.(im * wn)
    Be, De = fdestim_bd(zn, Y, U, A, C; type = Complex)
    fde = fresp(z, A, Be, C, De)

    @test computerr(fd, fde) < TOL
end


function unit_test_fdestim_cd(fset, m, n, p)
    A, B, C, D = buildss(n, m, p)
    w = 2pi * fset
    z = ztrans(fset)
    fd = fresp(z, A, B, C, D)
    U, Y, wn = ffdata2fddata(fd, w)
    zn = exp.(im * wn)
    Ce, De, resid = fdestim_cd(zn, Y, U, A, B)
    fde = fresp(z, A, B, Ce, De)

    @test computerr(fd, fde) < TOL
end


function unit_test_fdestim_cd_no_d(fset, m, n, p)
    A, B, C, D = buildss(n, m, p)
    D = zeros(p,m)
    w = 2pi * fset
    z = ztrans(fset)
    fd = fresp(z, A, B, C, D)
    U, Y, wn = ffdata2fddata(fd, w)
    zn = exp.(im * wn)
    Ce, De, resid = fdestim_cd(zn, Y, U, A, B, estimd=false)
    fde = fresp(z, A, B, Ce, De)

    @test computerr(fd, fde) < TOL
end


function unit_test_fdestim_cd_cmplx(fset, m, n, p)
    A, B, C, D = buildss(n, m, p) .+ im .* buildss(n, m, p)
    w = 2pi * fset
    z = ztrans(fset)
    fd = fresp(z, A, B, C, D)
    U, Y, wn = ffdata2fddata(fd, w)
    zn = exp.(im * wn)
    Ce, De, resid = fdestim_cd(zn, Y, U, A, B, type=Complex)
    fde = fresp(z, A, B, Ce, De)

    @test computerr(fd, fde) < TOL
end


function unit_test_ltifr_slow(fset, m, n, p)
    z = ztrans(fset)
    dd = zeros(Complex, n)
    T = randn(n, n)
    _, s, = svd(T)
    T = s[n] < 1e-6 ? T + 1e-6*I : T
    B = randn(n, m)

    nw = length(z)
    adiag = randn(n)
    A = diagm(0 => adiag)
    fkern1 = Array{Complex}(undef, nw, n, m)

    for fidx in 1:nw
        emad = z[fidx]*ones(n)
        dd = 1 ./ (emad-adiag)
        # for didx in 1:n
        #     dd[didx] = 1/(emad[didx]-adiag[didx])
        # end

        fkern1[fidx, :, :] = T * (diagm(0 => dd) * B)
    end
    Ae = T * (A/T)
    Be = T * B
    fkern = ltifr(Ae, Be, z) #SLOW?
    fkernf = ltifr(Ae, Be, z)
    @test computerr(fkern1, fkern) < TOL
    @test computerr(fkern1, fkernf) < TOL
end


function unit_test_ltifr_def(fset, m, n, p)
    A = [0 0; 1 0]
    B = randn(n, m)
    z = ztrans(fset .+ 1.0e-5)
    fkern = ltifr(A, B, z) #SLOW?
    fkernf = ltifr(A, B, z)

    @test computerr(fkernf, fkern) < TOL
end


function unit_test_fconv(fset, m, n, p)
    N = 100
    T = 100
    fset = -(N-1):N-1
    wc = fset
    wd = cf2df(wc, T)
    w = df2cf(wd, T)

    @test computerr(wc, w) < TOL
end


function unit_test_fdsid(fset, m, n ,p)
    N = length(fset)
    A, B, C, D = buildss(n, m, p)
    lam, = eigen(A)
    rho = maximum(abs.(lam))

    # create a random stable DT system
    A = A/rho/1.01
    w = 2pi * fset
    z = ztrans(fset)
    fd = fresp(z, A, B, C, D)

    u = randn(N, m)
    y, x = lsim((A, B, C, D), u; type = Real)

    yf = fft(y, 1)
    uf = fft(u, 1)
    fddata = (w, yf, uf)
    Ae, Be, Ce, De, xt, s = fdsid(fddata, n, 2n; estTrans = true, type = Real)
    fde = fresp(z, Ae, Be, Ce, De)

    @test computerr(fd, fde) < TOL
end


function unit_test_fdsid_no_d(fset, m, n ,p)
    N = length(fset)
    A, B, C, D = buildss(n, m, p)
    D = zeros(p,m)
    lam, = eigen(A)
    rho = maximum(abs.(lam))

    # create a random stable DT system
    A = A/rho/1.01
    w = 2pi * fset
    z = ztrans(fset)
    fd = fresp(z, A, B, C, D)

    u = randn(N, m)
    y, x = lsim((A, B, C, D), u; type = Real)

    yf = fft(y, 1)
    uf = fft(u, 1)
    fddata = (w, yf, uf)
    Ae, Be, Ce, De, xt, s = fdsid(fddata, n, 2n; estTrans = true, type = Real, estimd = false)
    fde = fresp(z, Ae, Be, Ce, De)

    @test computerr(fd, fde) < TOL
end


function unit_test_fdsid_cmplx(fset, m, n ,p)
    N = length(fset)
    A, B, C, D = buildss(n, m, p)
    lam, = eigen(A)
    rho = maximum(abs.(lam))

    # create a random stable DT system
    A = A/rho/1.01
    B = B + im*randn(n, m)
    D = D + im*randn(p, m)
    w = 2pi * fset
    z = ztrans(fset)
    fd = fresp(z, A, B, C, D)
    u = randn(N, m)
    y, x = lsim((A, B, C, D), u; type = Complex)
    yf = fft(y, 1)
    uf = fft(u, 1)
    fddata = (w, yf, uf)
    Ae, Be, Ce, De, xt, s = fdsid(fddata, n, 2*n; estTrans = true, type = Complex)
    fde = fresp(z, Ae, Be, Ce, De)

    @test computerr(fd, fde) < TOL
end


function unit_test_ffsid(fset, m, n, p)
    z = ztrans(fset)
    w = 2pi*fset
    A, B, C, D = buildss(n, m, p)
    fd = fresp(z, A, B, C, D)
    Ae, Be, Ce, De, s = ffsid(w, fd, n, n+1; type = Real, estimd = true)
    fde = fresp(z, Ae, Be, Ce, De)

    @test computerr(fd, fde) < TOL
end


function unit_test_ffsid_no_d(fset, m, n, p)
    z = ztrans(fset)
    w = 2pi*fset
    A, B, C, D = buildss(n, m, p)
    D = zeros(p,m)
    fd = fresp(z, A, B, C, D)
    Ae, Be, Ce, De, s = ffsid(w, fd, n, n+1; type = Real, estimd = false)
    fde = fresp(z, Ae, Be, Ce, De)

    @test computerr(fd, fde) < TOL
end


function unit_test_ffsid_complex(fset, m, n, p)
    z = ztrans(fset)
    w = 2pi*fset
    A, B, C, D = buildss(n, m, p)
    A = A + im*randn(n, n)
    B = B + im*randn(n, m)
    D = D + im*randn(p, m)

    fd = fresp(z, A, B, C, D)
    Ae, Be, Ce, De, s = ffsid(w, fd, n, n+1; type = Complex, estimd = true)
    fde = fresp(z, Ae, Be, Ce, De)

    @test computerr(fd, fde) < TOL
end


function unit_test_bilinear(fset, m, n, p)
    z = ztrans(fset)
    A, B, C, D = buildss(n, m, p)
    a, b, c, d = bilinear_c2d((A, B, C, D), 2)
    Ae, Be, Ce, De = bilinear_d2c((a, b, c, d), 2)

    @test (computerr(A, Ae) + computerr(B, Be) + computerr(C, Ce) + computerr(D, De)) < TOL
end


function unit_test_lrm(fset, m, n, p)
    N = length(fset)

    A, B, C, D = buildss(n, m, p)

    # create a random stable DT system
    lam = eigen(A).values
    rho = maximum(abs.(lam))
    A = A/rho/1.01

    z = ztrans(fset)

    fd = fresp(z, A, B, C, D)
    u = randn(N, m)
    y, x = lsim((A, B, C, D), u; type = Real)

    ff = lrm(u, y)

    @test computerr(fd, ff) < TOL
end


function unit_test_moebius(fset, m, n, p)
    N = length(fset)

    sset = randn(N) + im*randn(N)
    alpha, beta, gamma, delta = randn(4) + im*randn(4)
    zset = moebius_arg_inv(sset; alpha, beta, gamma, delta)
    sset1 = moebius_arg(zset; alpha, beta, gamma, delta)

    @test computerr(sset, sset1) < TOL

    A, B, C, D = buildss(n, m, p)
    zset = randn(N) + im*randn(N)
    alpha, beta, gamma, delta = randn(4) + im*randn(4)
    sset = moebius_arg(zset; alpha, beta, gamma, delta)
    fd = fresp(sset, A, B, C, D)
    Ae, Be, Ce, De = moebius((A, B, C, D); alpha, beta, gamma, delta)
    fde = fresp(zset, Ae, Be, Ce, De)

    @test computerr(fd, fde) < TOL

    sset = randn(N) + im*randn(N)
    zset = moebius_arg_inv(sset; alpha, beta, gamma, delta)
    fd = fresp(zset, A, B, C, D)
    Ae, Be, Ce, De = moebius_inv((A, B, C, D); alpha, beta, gamma, delta)
    fde = fresp(sset, Ae, Be, Ce, De)

    @test computerr(fd, fde) < TOL

    Ae, Be, Ce, De = moebius((A, B, C, D); alpha, beta, gamma, delta)
    a, b, c, d = moebius_inv((Ae, Be, Ce, De); alpha, beta, gamma, delta)
    sys = (a, b, c, d)
    sys0 = (A, B, C, D)

    err = 0
    for idx = 1:4
        err += computerr(sys[idx], sys0[idx])
    end
    @test err < TOL

end


function unit_test_markov_kung(fset, m, n, p)
    a, b, c, d = buildss(n, m, p)
    mp = markov((a, b, c), 2n)
    ae, be, ce = kung_realization(mp, n)
    me = markov((ae, be, ce), 2n)
    @test computerr(me, mp) < TOL
end


# Main test function
function unit_test(testfunc)
    N = 100
    if string(testfunc) == "unit_test_fresp_def" || string(testfunc) == "unit_test_ltifr_def"
        nmpset = [(2, 4, 12), (2, 3, 6)]

    elseif string(testfunc) == "unit_test_ltifr_slow"
        nmpset = [(4, 1, 1), (1,1,1), (2, 1, 1), (2, 4, 12), (2, 3, 12)] # change from (1,1,1) to (2,1,1)

    elseif string(testfunc) ==  "unit_test_fconv"
        nmpset = [(1, 1, 1)]

    elseif string(testfunc) == "unit_test_lrm"
        N = 3000
        nmpset = [(4, 1, 1), (1, 1, 1), (2, 4, 12)]

    elseif string(testfunc) == "unit_test_markov_kung"
        nmpset = [(4, 1, 1), (1, 1, 1), (4, 2, 1), (2, 4, 12)]

    else
        nmpset = [(4, 1, 1), (1, 1, 1), (2, 4, 12)]

    end
    fset = buildfset(N)

    for (n, m, p) in nmpset
        testfunc(fset, m, n, p)
    end

    return nothing
end


# Run the unit tests
@testset "fsid.jl" begin
    unit_test(unit_test_estimate_cd_1)
    unit_test(unit_test_estimate_cd_2)
    unit_test(unit_test_estimate_cd_3)
    unit_test(unit_test_estimate_bd)
    unit_test(unit_test_estimate_bd_2)
    unit_test(unit_test_estimate_bd_3)
    unit_test(unit_test_transpose_ffdata)
    unit_test(unit_test_fresp)
    unit_test(unit_test_fresp_def)
    unit_test(unit_test_fdestim_bd)
    unit_test(unit_test_fdestim_bd_no_d)
    unit_test(unit_test_fdestim_bd_cmplx)
    unit_test(unit_test_fdestim_cd)
    unit_test(unit_test_fdestim_cd_no_d)
    unit_test(unit_test_fdestim_cd_cmplx)
    unit_test(unit_test_ltifr_slow)
    unit_test(unit_test_ltifr_def)
    unit_test(unit_test_fconv)
    unit_test(unit_test_fdsid)
    unit_test(unit_test_fdsid_no_d)
    unit_test(unit_test_fdsid_cmplx)
    unit_test(unit_test_ffsid)
    unit_test(unit_test_ffsid_no_d)
    unit_test(unit_test_ffsid_complex)
    unit_test(unit_test_bilinear)
    unit_test(unit_test_lrm)
    unit_test(unit_test_moebius)
    unit_test(unit_test_markov_kung)
end
