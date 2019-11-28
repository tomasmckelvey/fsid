#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 17 11:10:38 2018

@author: mckelvey
"""
import numpy as np
from scipy import linalg
import fsid 


def example_ffsid():
    n = 2
    m = 3
    p = 4
    N = 100

    A = np.random.randn(n, n)
    B = np.random.randn(n, m)
    C = np.random.randn(p, n)
    D = np.random.randn(p, m)
    # Create frequency function
    fset = np.arange(0, N, dtype='float')/N
    w = 2*np.pi * fset
    wexp = np.exp(1j*2 * np.pi * fset)
    fd = fsid.fresp(wexp, A, B, C, D)
  
    # Estimate ss model from ffdata
    Ae, Be, Ce, De, s = fsid.ffsid(w, fd, n, n*2)

    # Frequency response of estimated model
    fde = fsid.fresp(wexp, Ae, Be, Ce, De)
    print('example_ffsid(): || G-Ge ||/||G|| = ', linalg.norm(fd-fde)/linalg.norm(fd))


def example_fdsid():
    import matplotlib.pyplot as plt

    n = 2
    m = 3
    p = 4
    N = 100

    A = np.random.randn(n, n)
    lam = linalg.eig(A)[0]
    rho = np.max( np.abs(lam)) 
    ## Here we create a random stable DT system
    A = A/rho/1.01
    B = np.random.randn(n, m)
    C = np.random.randn(p, n)
    D = np.random.randn(p, m)
    ## random excitation signal
    u = np.random.randn(N, m)
    ## Time domain simulation 
    y = fsid.lsim((A, B, C, D), u)
     
    plt.plot(y)
    plt.show()

    ## Crfeate the N point DFT of the signals        
    yf = np.fft.fft(y,axis=0)
    uf = np.fft.fft(u,axis=0)
    fset = np.arange(0, N, dtype='float')/N
    w = 2*np.pi * fset
    wexp = np.exp(1j*2 * np.pi * fset)

    fddata = (w, yf, uf)

    Ae, Be, Ce, De, xt, s =  fsid.fdsid(fddata, n, 2*n, estTrans=True)
    print('singular vales=', s)
    fde = fsid.fresp(wexp, Ae, Be, Ce, De)
    fd = fsid.fresp(wexp, A, B, C, D)

    print('With estTrans=true')
    print('|| G-Ge ||/||G|| = ', linalg.norm(fd-fde)/linalg.norm(fd))

    Ae1, Be1, Ce1, De1, xt1, s =  fsid.fdsid(fddata, n, 2*n, estTrans=False)
    print('singular vales=', s, ' model order n=', n)
    fde = fsid.fresp(wexp, Ae, Be, Ce, De)
    fd = fsid.fresp(wexp, A, B, C, D)

    print('|| G-Ge ||/||G|| = ', linalg.norm(fd-fde)/linalg.norm(fd))


def example_1():
    n = 2
    m = 3
    p = 4
    N = 100

    A = np.random.randn(n, n)
    lam = linalg.eig(A)[0]
    rho = np.max( np.abs(lam)) 
    ## Here we create a random stable DT system
    A = A/rho/1.01
    B = np.random.randn(n, m)
    C = np.random.randn(p, n)
    D = np.random.randn(p, m)
    ## random excitation signal
    u = np.random.randn(N, m)
    ## Time domain simulation 
    y = fsid.lsim((A, B, C, D), u)
     
    #plt.plot(y)
    #plt.show()

    ## Crfeate the N point DFT of the signals        
    yf = np.fft.fft(y,axis=0)
    uf = np.fft.fft(u,axis=0)
    fset = np.arange(0, N, dtype='float')/N
    w = 2*np.pi * fset
    wexp = np.exp(1j*2 * np.pi * fset)

    fddata = (w, yf, uf)
    # Test estimation of B D and Xt
    Be, De, xt, resid2  = fsid.fdestim_bd(wexp, yf, uf, A, C, estTrans=True, dtype='float')

    print('fdestim_bd and xt residual = ', resid2)

    ## Check that the frequency functions coincide
    fde = fsid.fresp(wexp, A, Be, C, De)
    fd = fsid.fresp(wexp, A, B, C, D)
    print('|| H-He ||/||H|| = ', linalg.norm(fd-fde)/linalg.norm(fd))

    xt = B[:,0]

    yy = fsid.fdsim((A,B,C,D), uf, wexp, xt)
    Be, De, xte, resid1  = fsid.fdestim_bd(wexp, yy, uf, A, C, estTrans=True)
    print('fdestim_bd residual = ', resid1)
    print('fdestim_bd residual xt= ', linalg.norm(xt-xte)/linalg.norm(xt))

    Ce, De, resid1  = fsid.fdestim_cd(wexp, yy, uf, A, B, xt, estTrans=True)
#    Be, De, resid1  = fdestim_bd(wexp, yy, uf, A, C, estTrans=False)
    print('fdestim_cd residual = ', resid1)



    fddata = (w, yf, uf)

    Ae, Be, Ce, De, xt, s =  fsid.fdsid(fddata, n, 2*n, estTrans=True)
#    Ae, Be, Ce, De, xt, s =  fsid.fdsid(fddata, n, 2*n, estTrans=False)
    print('singular vales=', s)
    fde = fsid.fresp(wexp, Ae, Be, Ce, De)
    fd = fsid.fresp(wexp, A, B, C, D)

    print('|| H-He ||/||H|| = ', linalg.norm(fd-fde)/linalg.norm(fd))



def example_2():
    import matplotlib.pyplot as plt

    n = 2
    m = 3

    p = 4
    N = 100

    
    N = 100    
    u = np.random.randn(N, m)
    A = np.random.randn(n, n)
    lam = linalg.eig(A)[0]
    rho = np.max( np.abs(lam)) 
    A = A/rho/1.01
    B = np.random.randn(n, m)
    C = np.random.randn(p, n)
    D = np.random.randn(p, m)
    y = fsid.lsim((A, B, C, D), u)
     
    plt.plot(y)
    plt.show()
     
    
    yf = np.fft.fft(y,axis=0)
    uf = np.fft.fft(u,axis=0)
    fset = np.arange(0, N, dtype='float')/N
    w = 2*np.pi * fset
    wexp = np.exp(1j*2 * np.pi * fset)

    fddata = (w, yf, uf)
    Be, De, xt, resid2  = fsid.fdestim_bd(wexp, yf, uf, A, C, estTrans=True, dtype='float')

    print('fdestim_bd residual = ', resid2)
    
    fde = fsid.fresp(wexp, A, Be, C, De)
    fd = fsid.fresp(wexp, A, B, C, D)

    print('|| H-He ||/||H|| = ', linalg.norm(fd-fde)/linalg.norm(fd))

    xt = B[:,0]

    yy = fsid.fdsim((A,B,C,D), uf, wexp, xt)
    Be, De, xt, resid1  = fsid.fdestim_bd(wexp, yy, uf, A, C, estTrans=True)
#    Be, De, resid1  = fdestim_bd(wexp, yy, uf, A, C, estTrans=False)
    print('fdestim_bd residual = ', resid1)

    Ce, De, resid1  = fsid.fdestim_cd(wexp, yy, uf, A, B, xt, estTrans=True)
#    Be, De, resid1  = fdestim_bd(wexp, yy, uf, A, C, estTrans=False)
    print('fdestim_cd residual = ', resid1)



    fddata = (w, yf, uf)

    Ae, Be, Ce, De, xt, s =  fsid.fdsid(fddata, n, 2*n, estTrans=True)
#    Ae, Be, Ce, De, s =  fdsid(fddata, n, 2*n, estTrans=False)
    print('singular vales=', s)
    fde = fsid.fresp(wexp, Ae, Be, Ce, De)
    fd = fsid.fresp(wexp, A, B, C, D)

    print('|| H-He ||/||H|| = ', linalg.norm(fd-fde)/linalg.norm(fd))
    

def example_ct_ffsid():
    n = 2
    m = 3
    p = 4
    N = 100
    T = 1.2e-1

    A = np.random.randn(n, n)
    B = np.random.randn(n, m)
    C = np.random.randn(p, n)
    D = np.random.randn(p, m)
    D = D*0
    # Create frequency function
    fset = np.arange(0, N, dtype='float')/N
    w = 2*np.pi * fset
#    wexp = np.exp(1j*2 * np.pi * fset)
    fd = fsid.fresp(1j*w, A, B, C, D)
  
    wd = fsid.cf2df(w,T)
    
    
    # Estimate ss model from ffdata
    Ae, Be, Ce, De, s = fsid.ffsid(wd, fd, n, n*2)

    # Convert to CT
    Aec, Bec, Cec, Dec =  fsid.bilinear_d2c((Ae, Be, Ce, De), T)
    # Frequency response of estimated model
    fde = fsid.fresp(1j*w, Aec, Bec, Cec, Dec)
    print('CT FF: || G-Ge ||/||G|| = ', linalg.norm((fd-fde)/linalg.norm(fd)))
 
    Ae, Be, Ce, De, s = fsid.ffsid(w, fd, n, n*2, CT=True, T=T, estimd=False)
    fde2 = fsid.fresp(1j*w, Ae, Be, Ce, De)
    print('CT2 FF: || G-Ge ||/||G|| = ', linalg.norm(fd-fde2)/linalg.norm(fd))
    print(De)
    

def example_ct_fdsid():
    n = 2
    m = 3
    p = 4
    N = 100
    T = 1.2e-1

    A = np.random.randn(n, n)
    B = np.random.randn(n, m)
    C = np.random.randn(p, n)
    D = np.random.randn(p, m)
    D = D*0
    print(D)
    # Create frequency function
    fset = np.arange(0, N, dtype='float')/N
    w = 2*np.pi * fset
#    wexp = np.exp(1j*2 * np.pi * fset)
    fd = fsid.fresp(1j*w, A, B, C, D)
    uf = np.random.randn(N, m) + 1j*np.random.randn(N, m)
   
    yf = fsid.fdsim((A ,B ,C ,D), uf, 1j*w)
  
    wd = fsid.cf2df(w,T)
    
    
    # Estimate ss model from ffdata
    Ae, Be, Ce, De, xt, s = fsid.fdsid((wd, yf, uf), n, n*2, estTrans=False, estimd=True)

    # Convert to CT
    Aec, Bec, Cec, Dec =  fsid.bilinear_d2c((Ae, Be, Ce, De), T)
    # Frequency response of estimated model
    fde = fsid.fresp(1j*w, Aec, Bec, Cec, Dec)
    print('CT FD: || G-Ge ||/||G|| = ', linalg.norm((fd-fde)/linalg.norm(fd)))
    print(Dec)
 
    Ae, Be, Ce, De, xt, s = fsid.fdsid((w, yf, uf), n, n*2, CT=True, T=T, estimd=False)
    fde2 = fsid.fresp(1j*w, Ae, Be, Ce, De)
    print('CT2 FD: || G-Ge ||/||G|| = ', linalg.norm(fd-fde2)/linalg.norm(fd))
    print(De)
 


example_ffsid()
example_fdsid()
example_1()
example_2()
example_ct_ffsid()
example_ct_fdsid()
