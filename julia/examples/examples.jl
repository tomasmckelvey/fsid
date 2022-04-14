using LinearAlgebra, FFTW, Plots


using fsid:
    estimate_cd,
    estimate_bd,
    transpose_ffdata,
    fresp,
    ffdata2fddata,
    fdestim_bd,
    fdestim_cd,
    fdsid,
    ffsid,
    bilinear_c2d,
    bilinear_d2c,
    lrm,
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


"""
    example_ffsid_rank()


"""
function example_ffsid_rank()
    n = 2
    m = 3
    p = 4
    N = 100

    A, B, C, D = buildss(n, m, p)
    fset = buildfset(N)/2
    w = 2pi*fset
    z = ztrans(fset)

    ffdata = fsid.fresp(z, A, B, C, D)
    Ae, Be, Ce, De, s = fsid.gffsid(z, ffdata, n, 2*n, type = Complex, estimd = true)
    fde = fsid.fresp(z, Ae, Be, Ce, De)
    err = computerr(ffdata, fde)
    println("||H - He||/||H|| = ", err)

    z2 = vcat(z, conj.(z))
    ffdata2 = vcat(ffdata, conj.(ffdata))
    Ae1, Be1, Ce1, De1, s = fsid.gffsid(z2, ffdata2, n, 2*n, type = Complex, estimd = true)
    fde = fsid.fresp(z2, Ae1, Be1, Ce1, De1)
    err = computerr(ffdata2, fde)
    println("||H - He||/||H|| = ", err)

    #return nothing

    A, B, C, D = buildss(n, m, p)

    # create frequency function
    fset = buildfset(N)
    w = 2pi * fset
    wexp = ztrans(fset)
    fd = fsid.fresp(wexp, A, B, C, D)

    # estimate ss model from ffdata
    Ae, Be, Ce, De, s = fsid.ffsid(w, fd, n, n*2)

    # frequency response of estimated model
    fde = fsid.fresp(wexp, Ae, Be, Ce, De)
    println("example_ffsid_rank(): ||G - Ge||/||G|| = ", computerr(fd, fde))

    return nothing
end


"""
    example_ffsid()


"""
function example_ffsid()
    n = 2
    m = 3
    p = 4
    N = 100

    A, B, C, D = buildss(n, m, p)

    # create frequency function
    fset = buildfset(N)
    w = 2pi * fset
    wexp = ztrans(fset)
    fd = fsid.fresp(wexp, A, B, C, D)

    # estimate ss model from ffdata
    Ae, Be, Ce, De, s = fsid.ffsid(w, fd, n, n*2)

    # frequency response of estimated model
    fde = fsid.fresp(wexp, Ae, Be, Ce, De)
    println("example_ffsid(): ||G - Ge||/||G|| = ", computerr(fd, fde))
end


"""
    example_fdsid()


"""
function example_fdsid()
    n = 2
    m = 3
    p = 4
    N = 100

    A, B, C, D = buildss(n, m, p)

    # create a random stable DT system
    lam = eigen(A).values
    rho = maximum(abs.(lam))
    A = A/rho/1.01

    # random excitation signal
    u = randn(N, m)

    # time domain simulation
    y = fsid.lsim((A, B, C, D), u)

    #plot(y)

    ## Crfeate the N point DFT of the signals
    yf = fft(y[1], 1)
    uf = fft(u, 1)
    fset = buildfset(N)
    w = 2pi * fset
    wexp = ztrans(fset)

    fddata = (w, yf, uf)

    Ae, Be, Ce, De, xt, s =  fsid.fdsid(fddata, n, 2*n, estTrans = true)
    # println("Singular vales = $s")
    fde = fsid.fresp(wexp, Ae, Be, Ce, De)
    fd = fsid.fresp(wexp, A, B, C, D)

    println("With estTrans = true")
    println("||G - Ge||/||G|| = ", computerr(fd, fde))

    Ae1, Be1, Ce1, De1, xt1, s =  fsid.fdsid(fddata, n, 2*n, estTrans = false)
    # println("Singular vales = $s, model order n = $n")
    fde = fsid.fresp(wexp, Ae, Be, Ce, De)
    fd = fsid.fresp(wexp, A, B, C, D)

    println("||G - Ge||/||G|| = ", computerr(fd, fde))
end


"""
    example_1()


"""
function example_1()
    n = 2
    m = 3
    p = 4
    N = 100

    A, B, C, D = buildss(n, m, p)

    # create a random stable DT system
    lam = eigen(A).values
    rho = maximum(abs.(lam))
    A = A/rho/1.01

    # random excitation signal
    u = randn(N, m)

    # time domain simulation
    y = fsid.lsim((A, B, C, D), u)

    #plt.plot(y)

    ## Crfeate the N point DFT of the signals
    yf = fft(y[1], 1)
    uf = fft(u, 1)
    fset = buildfset(N)
    w = 2pi * fset
    wexp = ztrans(fset)

    fddata = (w, yf, uf)
    # Test estimation of B D and Xt
    Be, De, xt, resid2  = fsid.fdestim_bd(wexp, yf, uf, A, C, estTrans = true, type = Real)

    # println("fdestim_bd and xt residual = ", resid2)

    ## Check that the frequency functions coincide
    fde = fsid.fresp(wexp, A, Be, C, De)
    fd = fsid.fresp(wexp, A, B, C, D)
    println("||H - He||/||H|| = ", computerr(fd, fde))

    xt = B[:, 1]

    yy = fsid.fdsim((A,B,C,D), uf, wexp, xt)

    Be, De, xte, resid1  = fsid.fdestim_bd(wexp, yy[1], uf, A, C; estTrans = true)
    # println("fdestim_bd residual = ", resid1)
    println("fdestim_bd residual xt = ", computerr(xt, xte))

    #Ce, De, resid1  = fsid.fdestim_cd(wexp, yy[1], uf, A, B, xt; estTrans = true)
    Ce, De, resid1  = fsid.fdestim_cd(wexp, yy[1], uf, A, B, xt)
#    Be, De, resid1  = fdestim_bd(wexp, yy, uf, A, C, estTrans = false)
    # println("fdestim_cd residual = ", resid1)

    fddata = (w, yf, uf)

    Ae, Be, Ce, De, xt, s =  fsid.fdsid(fddata, n, 2n, estTrans = true)
#    Ae, Be, Ce, De, xt, s =  fsid.fdsid(fddata, n, 2*n, estTrans = false)
    # println("Singular vales = ", s)
    fde = fsid.fresp(wexp, Ae, Be, Ce, De)
    fd = fsid.fresp(wexp, A, B, C, D)

    println("||H - He||/||H|| = ", computerr(fd, fde))
end


"""
    example_2()


"""
function example_2()
    n = 2
    m = 3
    p = 4
    N = 100

    A, B, C, D = buildss(n, m, p)

    # create a random stable DT system
    lam = eigen(A).values
    rho = maximum(abs.(lam))
    A = A/rho/1.01

    u = randn(N, m)

    y = fsid.lsim((A, B, C, D), u)

    plot(y)

    yf = fft(y[1], 1)
    uf = fft(u, 1)
    fset = buildfset(N)
    w = 2pi * fset
    wexp = ztrans(fset)

    fddata = (w, yf, uf)

    Be, De, xt, resid2 = fsid.fdestim_bd(wexp, yf, uf, A, C; estTrans = true, type = Real)
    # println("fdestim_bd residual = ", resid2)

    fde = fsid.fresp(wexp, A, Be, C, De)
    fd = fsid.fresp(wexp, A, B, C, D)
    println("||H - He||/||H|| = ", computerr(fd, fde))

    xt = B[:, 1]

    yy = fsid.fdsim((A,B,C,D), uf, wexp, xt)
    Be, De, xt, resid1  = fsid.fdestim_bd(wexp, yy[1], uf, A, C; estTrans = true)
#    Be, De, resid1  = fdestim_bd(wexp, yy, uf, A, C, estTrans = false)
    # println("fdestim_bd residual = ", resid1)

    #Ce, De, resid1  = fsid.fdestim_cd(wexp, yy, uf, A, B, xt; estTrans = true)
    Ce, De, resid1  = fsid.fdestim_cd(wexp, yy[1], uf, A, B, xt)
#    Be, De, resid1  = fdestim_bd(wexp, yy, uf, A, C, estTrans = false)
    # println("fdestim_cd residual = ", resid1)

    fddata = (w, yf, uf)

    Ae, Be, Ce, De, xt, s =  fsid.fdsid(fddata, n, 2*n, estTrans = true)
#    Ae, Be, Ce, De, s =  fdsid(fddata, n, 2*n, estTrans = false)
    # println("Singular vales = ", s)
    fde = fsid.fresp(wexp, Ae, Be, Ce, De)
    fd = fsid.fresp(wexp, A, B, C, D)

    println("||H - He||/||H|| = ", computerr(fd, fde))
end


"""
    example_ct_ffsid()


"""
function example_ct_ffsid()
    n = 2
    m = 3
    p = 4
    N = 100
    T = 1.2e-1

    A, B, C, D = buildss(n, m, p)
    D = D*0

    # create frequency function
    fset = buildfset(N)
    w = 2pi * fset
#    wexp = ztrans(fset)
    fd = fsid.fresp(im*w, A, B, C, D)

    wd = fsid.cf2df(w,T)

    # estimate ss model from ffdata
    Ae, Be, Ce, De, s = fsid.ffsid(wd, fd, n, n*2)

    # convert to CT
    Aec, Bec, Cec, Dec =  fsid.bilinear_d2c((Ae, Be, Ce, De), T)

    # frequency response of estimated model
    fde = fsid.fresp(im*w, Aec, Bec, Cec, Dec)
    println("CT FF: ||G - Ge||/||G|| = ", computerr(fd, fde))

    Ae, Be, Ce, De, s = fsid.ffsid(w, fd, n, n*2, CT = true, T = T, estimd = false)
    fde2 = fsid.fresp(im*w, Ae, Be, Ce, De)
    println("CT2 FF: ||G - Ge||/||G|| = ", computerr(fd, fde2))
    # println(De)
end


"""
    example_ct_fdsid()


"""
function example_ct_fdsid()
    n = 2
    m = 3
    p = 4
    N = 100
    T = 1.2e-1

    A, B, C, D = buildss(n, m, p)
    D = D*0
    # println(D)
    # Create frequency function
    fset = buildfset(N)
    w = 2pi * fset
#    wexp = ztrans(fset)
    fd = fsid.fresp(im*w, A, B, C, D)
    uf = randn(N, m) + im*randn(N, m)

    yf = fsid.fdsim((A ,B ,C ,D), uf, im*w)

    wd = fsid.cf2df(w,T)

    # estimate ss model from ffdata
    Ae, Be, Ce, De, xt, s = fsid.fdsid((wd, yf[1], uf), n, 2n; estTrans = false, estimd = true)

    # convert to CT
    Aec, Bec, Cec, Dec =  fsid.bilinear_d2c((Ae, Be, Ce, De), T)

    # frequency response of estimated model
    fde = fsid.fresp(im*w, Aec, Bec, Cec, Dec)
    println("CT FD: ||G - Ge||/||G|| = ", computerr(fd, fde))
    # println(Dec)

    Ae, Be, Ce, De, xt, s = fsid.fdsid((w, yf[1], uf), n, 2n; CT = true, T = T, estimd = false)
    fde2 = fsid.fresp(im*w, Ae, Be, Ce, De)
    println("CT2 FD: ||G - Ge||/||G|| = ", computerr(fd, fde2))
    # println(De)
end


"""
    example_ffsid_lrm()


"""
function example_ffsid_lrm()
    N = 2000
    nmpset = [(4, 1, 1), (1, 1, 1), (2, 4, 12)]
    nmpset = [(10, 5, 5)]
    for (n, m, p) in nmpset
        A, B, C, D = buildss(n, m, p)

        # create a random stable DT system
        lam = eigen(A).values
        rho = maximum(abs.(lam))
        A = A/rho/1.01

        fset = buildfset(N)
        z = ztrans(fset)
        fd = fresp(z, A, B, C, D)
        u = randn(N, m)
        y, x = lsim((A, B, C, D), u, type = Real)
        ff = lrm(u, y, n = 2, Nw = 30)
        # yf = np.fft.fft(y, axis=0)
        # uf = np.fft.fft(u, axis=0)

        println("||fd - ff||/||fd|", computerr(fd, ff))
        #plot(20log10.(abs.(fd[:, 1, 1])))
        #plot(20log10.(abs.(ff[:, 1, 1])))
        #plot(20log10.(abs.(ff[:, 1, 1] - fd[:, 1, 1])))
    end

    N = 2000
    A = [0.9999 1; 0 .9]
    B = [0; 1][:, :]
    C = [1 0]
    D = [0][:, :]
    u = randn(N, 1)
    y, x = lsim((A, B, C, D), u; type = Real)

    fset = buildfset(N)
    z = ztrans(fset)
    fd = fresp(z, A, B, C, D)
    u = randn(N, 1)
    y, x = lsim((A, B, C, D), u; type = Real)
    ff = lrm(u, y; n=1, Nw=3)
    # yf = np.fft.fft(y, axis=0)
    # uf = np.fft.fft(u, axis=0)

    println("||fd - ff||/||fd|", computerr(fd, ff))
    #plot(20log10.(abs.(fd[1:20, 1, 1])))
    #plot(20log10.(abs.(ff[1:20, 1, 1])))
    #plot(20log10.(abs.(ff[1:20, 1, 1] - fd[1:20, 1, 1])))
end


"""
    examples()


"""
function examples()
    example_ffsid()
    example_fdsid()
    example_1()
    example_2()
    example_ct_ffsid()
    example_ct_fdsid()
    example_ffsid_rank()
    example_ffsid_lrm()
end