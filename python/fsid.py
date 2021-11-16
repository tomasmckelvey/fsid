#!/usr/bin/env python2estimd
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  9 08:00:56 2017

@author: Tomas McKelvey
"""

import numpy as np
from scipy import linalg

def fsid():
    """
    This is the FSID toolbox for Matlab version 1.3, 2021-07-14
    
    Functions: fdsid, gfdsid, ffsid, gffsid, lrm 

    FSID is open source software licenced under the GNU GPL v.3 
    FSID can be accessed at https://github.com/tomasmckelvey/fsid

    """
    print("FSID for python version 1.3, 2021-07-14")


def lrm(u, y, n=2 , Nw=None, DFTindices=None, lam=0):
    """
        Calulates the non-parametric frequency MIMO function estimate using the Local Rational Model mathod.

    Parameters
    ----------
    u : array
        input samples where element u[i,j] holds the sample for time index i and input channel j 
    y : array 
        output samples where element y[i,j] holds the sample for time index i and input channel j 
    n : integer
        local rational model order (default n=2)
    Nw :integer (optional)
        determines local frequency window. To determine the estimate at DFT frequency index k,
        DFT samples from index k-Nw to k+Nw are employed (with wraparound at the edges)
        if Nw = None (default) then Nw Nw = 2*np.int(np.ceil(((m+2)*n+1.0)/2)) where m is the number of input channels 
    DFTindices : array of integers
        a subset of the DFT indices. The local model is inlu calculated for frequency indices in this set 
    lam : real valued scalar. (optinal) 
        if lam>0 then the LS-problem is augmented with parameter penalty scaled with lam*norm(R) 
        where R is the associated regression matrix.
        
    Returns
    -------
    ff : array of frequency function estimate where element ff[k,i,j] correspond to DFT frequency index k, 
        output index i and input index j. If DFTindices is given then ff[i,i,j] holds the frequency function 
        estimate at DFT frequency index DFTindices[i]

    """
    us = np.shape(u)
    if len(us) == 1:
        m = 1
        u = np.reshape(u,(us[0],1))       
    else:
        m = us[1]
    Nu = us[0]
    ys = np.shape(y)
    if len(ys) == 1:
        p = 1
        y = np.reshape(y,(ys[0],1))
    else:
        p = ys[1]
    Ny = ys[0]            
    if Nu != Ny:
        print("lrm: incompatible number of time samples in u and y")
        return False  
    if Nw is None:
        Nw = 2*int(np.ceil(((m+2)*n+1.0)/2))
    #print("Nw ",Nw)
    if Nw <int(np.ceil(((m+2)*n+1.0)/2)):
        print("Error: Nw too small.")
        return False
    yf = np.fft.fft(y, axis=0)
    uf = np.fft.fft(u, axis=0)
    R = np.vander(np.arange(-Nw,Nw+1),n+1)
#    print("NW", Nw)
#    print("size R", np.shape(R))
    yfe = np.vstack((yf[-Nw:,:], yf, yf[:Nw] ))
    ufe = np.vstack((uf[-Nw:,:], uf, uf[:Nw] ))
    iset = np.arange(-Nw,Nw+1)
    if DFTindices is None:
        DFTindices = np.arange(Ny)
#    for i in np.arange(Ny):
    ff = np.zeros((np.size(DFTindices), p, m), dtype=complex)
    for fi, i in enumerate(DFTindices):
        for pidx in range(p):
            # A(z) yf = B(z) uf + T(z)
            yy = yfe[Nw+i+iset,pidx]
 #           uu = ufe[Nw+i+iset,0]
            RR = np.hstack((R, np.matmul(np.diag(-yy), R[:,:n])  ))
            for midx in np.flip(range(m),0):
                RR = np.hstack((RR, np.matmul(np.diag(ufe[Nw+i+iset, midx]), R)))
            if lam != 0:
#                nc = np.size(RR,1)
#                print(m,n,(n+1)*m+n, np.size(RR,1))
                nc = (n+1)*(m+1)+n
                yy1 = np.concatenate((yy, np.zeros(nc)),0)
                RR1 = np.concatenate((RR, lam*linalg.norm(RR)*np.eye(nc)), 0)
                lsans = linalg.lstsq(RR1, yy1)
            else:
                lsans = linalg.lstsq(RR, yy)
            ht = lsans[0]
            ht = np.flip(ht,0)
            ff[fi,pidx,:] = ht[:(n+1)*m:(n+1)]
    #print(np.shape(RR))
    return ff
            
def kung_realization(mp, n, q=0):
    """
        Calulates the state-space realization `(a,b,c)` from Markov parameters 
        using Kung's relization algorithm' 

    Parameters
    ----------
    m : array of size(N,p,m) where N>=2n
        Array of Marov parameters m[i,j-1,k-1] hold row j 
        and column k of the Markov parameter of sample i  
    n : integer
        Model order state-space system.
    q : integer, optional
        Number of rows in Hankel matrix. The default is n+1.

    Returns
    -------
    tuple (a,b,c).

    """
    if q==0:
        q=n+1
    N = np.size(mp, 0)
    p = np.size(mp, 1)
    m = np.size(mp, 2)
    ncol = N-q+1
    if ncol<n:
        print("n, q and N are not compatible. N>=q+n-1 must be satisfied")
        return False
    H = np.empty((p*q, ncol*m), dtype=type(mp[0,0,0]))
    for j in range(ncol):
        for i in range(q):
            H[i*p:(i+1)*p, j*m:(j+1)*m] =mp[i+j, :, :]
    u, s, vh = linalg.svd(H, full_matrices=False, overwrite_a=True)
    c = u[:p, :n]
    lh = u[:p*(q-1), :n]
    rh = u[p:, :n]
    lsres = linalg.lstsq(lh, rh, overwrite_a=True, overwrite_b=True)
    a = lsres[0]
    b = np.dot(np.diag(s[:n]), vh[:n,:m]) 
    return a, b, c
    
def markov(sys, N):
    """
    Calculate markov parameters from sys = (a,b,c)

    Parameters
    ----------
    sys : tuple
        (a,b,c) state-space matrices.
    N : integer
        numer of Markov parameters to generate.

    Returns
    -------
    mp : array
        mp[i,:,;] is Markov parameter C(A^i)B.
    """    
    (a, b, c) = sys[:3]
    n = np.size(a, 0)
    m = np.size(b, 1)
    p = np.size(c, 0)
    mp = np.empty((N, p, m), dtype=type(a[0,0]))
    aa = np.eye(n, dtype=type(a[0,0]))
    for i in range(N):
        mp[i, :, :] = np.dot(c, np.dot(aa, b))
        aa = np.dot(aa, a)
    return mp

def make_sys_real(sys):
    """
    Convert realization sys into a real-valued realization 

    Parameters
    ----------
    sys : tuple
        sys = (a, b, c) or sys =  (a, b, c, d)

    Returns
    -------
    sysr : tuple
        sysr = (ar, br, cr, dr) the realization with real valued matrices

    """
    a, b, c = sys[:3]
    n = np.size(a,0)
    mp = markov(sys, 2*n )
    mpr = np.real(mp)
    a, b, c = kung_realization(mpr, n) 
    if len(sys)==3:
        return (a, b, c)
    else:
        return (a, b, c, np.real(sys[3]))

def make_obs_real(a,c):
    """
    Convert (a,c) into real-valued matrices by approximating a real valued
    observability range space to the original one

    Parameters
    ----------
    a : matrix
    b : matrix 

    Returns
    -------
    ar, br  : tuple
        the real valued matrices

    """
    n = np.size(a,0)
    p = np.size(c,0)
    obs = np.empty((p*(n+1), n), dtype='complex')
    obs[0:p, :] = c
    for i in range(n):
        obs[p*(i+1):p*(i+2), :] = np.dot(obs[p*i:p*(i+1), :], a)
    obsr = np.concatenate((np.real(obs), np.imag(obs)), 1)
    u, s, vh = linalg.svd(obsr, full_matrices=False, overwrite_a=True)
    c = u[:p, :n]
    lh = u[:p*n, :n]
    rh = u[p:, :n]
    lsres = linalg.lstsq(lh, rh, overwrite_a=True, overwrite_b=True)
    a = lsres[0]
    return a, c     
        
def ls_estim_cd(ffdata, z, a, b, dtype='float', estimd=True):
    """
    Estimates the c and d matrices given a, b matrices and frequency function data 'ffdata'.
    
    Calulates the c and d matrices for a linear dynamic system given the a and b 
    matrices and samples of frequency function data. It solves 
    
    min_{c,d} sum_i || d + c*inv(z[i]*I-a)*b - ffdata[i,:,:] ||^2_F

    if estimd == False 

    min_{c} sum_i || c*inv(z[i]*I-a)*b - ffdata[i,:,:] ||^2_F

    if dtype='float' a real valued solution is calulated. if dtype='complex' 
    the solution is complex valued
    
    Parameters
    ----------
    ffdata : matrix_like 
        frequency data packed in a matrix. fftata[i,:,:] is the frequency function 
        matrix corresponding to sample i
    z : matrix_like
        vector with the frequecy points to evaluate, i.e. for DT z = exp(j*w) where w is frequencies in 
        radians per sample
    a : matrix_like
        the a matrix
    b : matrix_like
        the b matrix
    dtype: optional
    estimd: optional
        data type of model either 'float' or 'complex'
        if set to False no d matrix is esimated and a zero d matrix is returned
        
    Returns 
    -------
    
    c : matrix_like
        the LS-optimal c matrix
    d : matrix_like
        the LS-optimal d matrix

    """
    m = np.size(b, 1)
    nw = np.size(ffdata, 0)
    fk = ltifr_slow(a, b, z)
    fkstack = np.concatenate(fk, 1) #regressor for c matrix
    if estimd:
        dreg = np.tile(np.eye(m), (1, nw)) #regressor for d matrix
        R = np.concatenate((dreg, fkstack)) #complete regressor
    else:
        R = fkstack    
    RH = np.concatenate(ffdata, 1) #right hand side in LS
    if dtype == 'float':
        RR = np.concatenate([np.real(R), np.imag(R)], 1)
        RRH = np.concatenate([np.real(RH), np.imag(RH)], 1)
        lsans = linalg.lstsq(RR.T, RRH.T, overwrite_a=True, overwrite_b=True)
    else:
        lsans = linalg.lstsq(R.T, RH.T, overwrite_a=True, overwrite_b=True)
    dc = lsans[0].T
    if estimd:    
        return dc[:, m:], dc[:, :m]              #return c and d separatly
    else:
        return dc, np.zeros((np.size(dc, 0), m), dtype)

def ls_estim_bd(ffdata, z, a, c, dtype='float', estimd=True):
    """ 
    Estimates the b and d matrices given a, c matrices and frequency function data 'ffdata'.
    
    Calulates the b and d matrices for a linear dynamic system given the a and b 
    matrices and samples of frequency function data. It solves 
    
    min_{b,d} sum_i || d + c*inv(z[i]*I-a)*b - ffdata[i,:,:] ||^2_F

    if estimd == False 

    min_{b} sum_i || c*inv(z[i]*I-a)*b - ffdata[i,:,:] ||^2_F
    
    if dtype='float' a real valued solution is calulated. if dtype='complex' 
    the solution is complex valued
    
    Parameters
    ----------
    ffdata: matrix_like 
        frequency data packed in a matrix. fftata[i,:,:] is the frequency function 
        matrix corresponding to sample i
    z : matrix_like
        vector with the frequecy points to evaluate, i.e. for DT z = exp(j*w) where w is frequencies in 
        radians per sample
    a: matrix_like
        the a matrix
    c: matrix_like
        the c matrix
    dtype: optional
        data type of model either 'float' or 'complex'
    estimd: optional
        if set to False no d matrix is esimated and a zero d matrix is returned

    Returns 
    -------
    
    b : matrix_like
        the LS-optimal b matrix
    d : matrix_like
        the LS-optimal d matrix
    """

    fdt = transp_ffdata(ffdata)
    bt, dt = ls_estim_cd(fdt, z, a.T, c.T, dtype, estimd)
    return bt.T, dt.T

def transp_ffdata(ffdata):
    """Transposes ffdata (changes inputs and outputs) """
    m = np.size(ffdata, 2)
    p = np.size(ffdata, 1)
    nw = np.size(ffdata, 0)
    fdt = np.empty([nw, m, p], dtype='complex')
    for idx in range(nw):
        fdt[idx, :, :] = ffdata[idx, :, :].T
    return fdt

def fresp_slow(z, a, b, c, d):
    """Frequency response of ss-model (a,b,c,d)
    calulated as fresp[i,:,:] = d+c*inv(I*z[i]-a)*b (slow version)"""
    n = np.size(a, 0)
    m = np.size(b, 1)
    p = np.size(c, 0)
    nw = np.size(z, 0)
    frsp = np.empty([nw, p, m], dtype='complex')
    for widx in range(nw):
        aa = np.eye(n) * z[widx] - a
        frsp[widx, :, :] = np.linalg.multi_dot([c, np.linalg.solve(aa, b)]) + d
    return frsp

def fresp(z, a, b, c, d, noWarning=False):
    """Frequency response of ss-model (a,b,c,d)
    calulated as fresp[i,:,:] = d+c*inv(I*z[i]-a)*b (fast version)"""
    n = np.size(a, 0)
    m = np.size(b, 1)
    p = np.size(c, 0)
    nw = np.size(z, 0)
    lam, t = linalg.eig(a)
    if np.linalg.matrix_rank(t) < n:
        if  not noWarning:
            print("Matrix a defective. Eigenvectors do not form a basis. Using slow mode.")
        return fresp_slow(z, a, b, c, d)
    else:
        it = linalg.inv(t)
        bb = np.dot(it, b)
        cc = np.dot(c, t)
        frsp = np.empty([nw, p, m], dtype='complex')
        for widx in range(nw):
            da = np.ones(n) * z[widx] - lam
            frsp[widx, :, :] = np.linalg.multi_dot([cc, np.diag(1/da), bb]) + d
        return frsp

def ltifr_slow(a, b, z):
    """Calculates the (nz, n, m) size frequency kernel 
      fkern[i,:,:] = inv(z[i]*eye(n)-a)*b (slow version)   
    """
    n = np.size(a, 0)
    m = np.size(b, 1)
    nw = len(z)
    fkern = np.empty([nw, n, m], dtype='complex')
    for widx in range(nw):
        fkern[widx, :, :] = np.linalg.solve(np.eye(n) * z[widx] - a, b)
    return fkern

def ltifr(a, b, z, noWarning=False):
    """Calculates the (nz,n,m) size frequency kernel
        fkern[i,:,:] = inv(z[i]*eye(n)-a)*b    
    """
    n = np.size(a, 0)
    m = np.size(b, 1)
    nw = len(z)
    lam, t = linalg.eig(a)
    if np.linalg.matrix_rank(t) < n:
        if not noWarning:
            print("Matrix a defective. Eigenvectors do not form a basis. Using slow mode.")
        return ltifr_slow(a, b, z)
    else:
        it = linalg.inv(t)
        bb = np.dot(it, b)
        fkern = np.empty([nw, n, m], dtype='complex')
        for widx in range(nw):
            da = np.ones(n) * z[widx] - lam
            fkern[widx, :, :] = np.linalg.multi_dot([t, np.diag(1/da), bb])    
        return fkern

def ffsid_old(w, ffdata, n, q, dtype='float', estimd=True, CT=False, T=1):
    """Estimate a state-space model (a,b,c,d) from frequency function data
    
    Parameters
    ==========
    w : array_like
        vector of frequencies in rad/sample [-pi,pi] or [0, 2pi]
        if CT = True unit it radians/s (-inf, +inf)
    ffdata : array_like
        matrix of frequency function data. ffdata[i,:,:] is the frequency response matrix
        at frequency w[i]
    n : integer
        the model order of the ss-model (a,b,c,d), i.e. a is a size (n x n) matrix
    q : integer
        the numer of block rows used in the intermediate matrix. Must satisfy q>n
    dtype : data type, optional
        if dtype = 'float' a real valued solution (a,b,c,d) is returned.
        if dtype = 'complex' a complex valued solution (a,b,c,d) is returned.
    estimd: optional
        if set to False no d matrix is esimated and a zero d matrix is returned
    CT : if set to true a continuous time (CT) model is esimated
         if set to false a discrete time (DT) model is esimated (default)
    T :  a frequency scaling factor for the bilinear transformation used when CT=True. 
        Default is 1. If CT=False parameter T is disregarded 
        
    Returns
    =======
    a  : matrix_like
        the estimated a matrix
    b  : matrix_like
        the estimated b matrix
    c  : matrix_like
        the estimated c matrix
    d  : matrix_like
        the estimated d matrix (or zero matrix if estimd=False)
    s  : matrix_like
        a vector of the singular values 
    
    
    """
    if CT:
        ad, bd, cd, dd, s  = ffsid(cf2df(w,T), ffdata, n, q, dtype, estimd=True, CT=False, T=T)
        a, be, ce, de = bilinear_d2c((ad, bd, cd, dd), T)
        if not estimd:
            be, de = ls_estim_bd(ffdata, 1j*w, a, ce, dtype, estimd)
            ce, de = ls_estim_cd(ffdata, 1j*w, a, be, dtype, estimd)
            be, de = ls_estim_bd(ffdata, 1j*w, a, ce, dtype, estimd)
            ce, de = ls_estim_cd(ffdata, 1j*w, a, be, dtype, estimd)
        return a, be, ce, de, s
            
    nwfd, p, m = np.shape(ffdata)
    nw = np.size(w, 0)
    z = np.exp(1j*w)
    if q < n+1:
        print("Error: q must satidfy q>n.")
        return False
    if nw != nwfd:
        print("Error: w and ffdata sizes does not match!")
        return False
    y = np.empty([p*q, nw*m], dtype='complex')
    u = np.empty([m*q, nw*m], dtype='complex')
    for widx in range(nw):
        y[:p, widx*m:(widx+1)*m] = ffdata[widx, :, :]
        u[:m, widx*m:(widx+1)*m] = np.eye(m)
        zx = z[widx]
        for qidx in range(q)[1:]:
            y[qidx*p:(qidx+1)*p, widx*m:(widx+1)*m] = zx*ffdata[widx, :, :]
            u[qidx*m:(qidx+1)*m, widx*m:(widx+1)*m] = zx*np.eye(m)
            zx *= z[widx]
    if dtype == 'float':
        hU = np.concatenate((np.real(u), np.imag(u)), 1)
        hY = np.concatenate((np.real(y), np.imag(y)), 1)
    else:
        hU = u
        hY = y
    h = np.concatenate((hU, hY))
    r = linalg.qr(h.T, mode='r', overwrite_a=True)  #Calulates the projection
    r = r[0].T
    r22 = r[m*q:, m*q:]
    u, s, vh = linalg.svd(r22, full_matrices=False, overwrite_a=True)
    c = u[:p, :n]
    lh = u[:p*(q-1), :n]
    rh = u[p:, :n]
    lsres = linalg.lstsq(lh, rh, overwrite_a=True, overwrite_b=True)
    a = lsres[0]
    be, de = ls_estim_bd(ffdata, z, a, c, dtype, estimd)
    ce, de = ls_estim_cd(ffdata, z, a, be, dtype, estimd)
    be, de = ls_estim_bd(ffdata, z, a, ce, dtype, estimd)
    ce, de = ls_estim_cd(ffdata, z, a, be, dtype, estimd)
    return a, be, ce, de, s


def ffsid(w, ffdata, n, q, dtype='float', estimd=True, CT=False, T=1):
    """Estimate a state-space model (a,b,c,d) from frequency function data
    
    Parameters
    ==========
    w : array_like
        vector of frequencies in rad/sample [-pi,pi] or [0, 2pi]
        if CT = True unit in radians/s (-inf, +inf)
    ffdata : array_like
        matrix of frequency function data. ffdata[i,:,:] is the frequency response matrix
        at frequency w[i]
    n : integer
        the model order of the ss-model (a,b,c,d), i.e. a is a size (n x n) matrix
    q : integer
        the numer of block rows used in the intermediate matrix. Must satisfy q>n
    dtype : data type, optional
        if dtype = 'float' a real valued solution (a,b,c,d) is returned.
        if dtype = 'complex' a complex valued solution (a,b,c,d) is returned.
    estimd: optional
        if set to False no d matrix is esimated and a zero d matrix is returned
    CT: if set to true a continuous time (CT) model is esimated
         if set to false a discrete time (DT) model is esimated (default)
    T :  a frequency scaling factor for the bilinear transformation used when CT=True. 
        Default is 1. If CT=False parameter T is disregarded 
        
    Returns
    =======
    a  : matrix_like
        the estimated a matrix
    b  : matrix_like
        the estimated b matrix
    c  : matrix_like
        the estimated c matrix
    d  : matrix_like
        the estimated d matrix (or zero matrix if estimd=False)
    s  : matrix_like
        a vector of the singular values 
    
    
    """
    if CT:
        ad, bd, cd, dd, s  = ffsid(cf2df(w,T), ffdata, n, q, dtype, estimd=True, CT=False, T=T)
        a, be, ce, de = bilinear_d2c((ad, bd, cd, dd), T)
        if not estimd:
            be, de = ls_estim_bd(ffdata, 1j*w, a, ce, dtype, estimd)
            ce, de = ls_estim_cd(ffdata, 1j*w, a, be, dtype, estimd)
            be, de = ls_estim_bd(ffdata, 1j*w, a, ce, dtype, estimd)
            ce, de = ls_estim_cd(ffdata, 1j*w, a, be, dtype, estimd)
        return a, be, ce, de, s
            
    z = np.exp(1j*w)
    return gffsid(z, ffdata, n, q, dtype=dtype, estimd=estimd)

def gffsid(z, ffdata, n, q, dtype='float', estimd=True):
    """Estimate a state-space model (a,b,c,d) from general frequency function data 
    by mininizing the Frobenius norm 
        sum_i   || ffdata[i,:,:] - d - c*inv(z[i]*eye(n)-A)*b ||^2_F

    Parameters
    ==========
    z : array_like
        vector of complex numbers
    ffdata : array_like
        matrix of frequency function data. ffdata[i,:,:] is the frequency response matrix
        at complex point z[i]
    n : integer
        the model order of the ss-model (a,b,c,d), i.e. a is a size (n x n) matrix
    q : integer
        the numer of block rows used in the intermediate matrix. Must satisfy q>n
    dtype : data type, optional
        if dtype = 'float' a real valued solution (a,b,c,d) is returned.
        if dtype = 'complex' a complex valued solution (a,b,c,d) is returned.
    estimd: optional
        if set to False no d matrix is esimated and a zero d matrix is returned
        
    Returns
    =======
    a  : matrix_like
        the estimated a matrix
    b  : matrix_like
        the estimated b matrix
    c  : matrix_like
        the estimated c matrix
    d  : matrix_like
        the estimated d matrix (or zero matrix if estimd=False)
    s  : matrix_like
        a vector of the singular values 
       
    """
            
    nwfd, p, m = np.shape(ffdata)
    nw = np.size(z, 0)
    if q < n+1:
        print("Error: q must satisfy q>n.")
        return False
    if nw != nwfd:
        print("Error: w and ffdata sizes does not match!")
        return False
    y = np.empty([p*q, nw*m], dtype='complex')
    if estimd:
        u = np.empty([m*q, nw*m], dtype='complex')
    else:
        u = np.empty([m*(q-1), nw*m], dtype='complex')
        
            
    for widx in range(nw):
        y[:p, widx*m:(widx+1)*m] = ffdata[widx, :, :]
        u[:m, widx*m:(widx+1)*m] = np.eye(m)
        zx = z[widx]
        for qidx in range(q)[1:]:
            y[qidx*p:(qidx+1)*p, widx*m:(widx+1)*m] = zx*ffdata[widx, :, :]
            if estimd or qidx<q-1:
                u[qidx*m:(qidx+1)*m, widx*m:(widx+1)*m] = zx*np.eye(m)
            zx *= z[widx]
    if dtype == 'float':
        hU = np.concatenate((np.real(u), np.imag(u)), 1)
        hY = np.concatenate((np.real(y), np.imag(y)), 1)
    else:
        hU = u
        hY = y
    h = np.concatenate((hU, hY))
    r = linalg.qr(h.T, mode='r', overwrite_a=True)  #Calulates the projection
    r = r[0].T
    if estimd:
        r22 = r[m*q:, m*q:]
    else:
        r22 = r[m*(q-1):, m*(q-1):]                    
    u, s, vh = linalg.svd(r22, full_matrices=False, overwrite_a=True)
    c = u[:p, :n]
    lh = u[:p*(q-1), :n]
    rh = u[p:, :n]
    lsres = linalg.lstsq(lh, rh, overwrite_a=True, overwrite_b=True)
    a = lsres[0]
    be, de = ls_estim_bd(ffdata, z, a, c, dtype, estimd)
    ce, de = ls_estim_cd(ffdata, z, a, be, dtype, estimd)
    be, de = ls_estim_bd(ffdata, z, a, ce, dtype, estimd)
    ce, de = ls_estim_cd(ffdata, z, a, be, dtype, estimd)
#    pip = np.eye(m*nw) - np.linalg.multi_dot( [u.T.conj(), np.linalg.inv( np.dot(u , u.T.conj())),  u]) 
#    x = np.dot(y, pip)
#    pip, u, R, Gproj = project_fdshape(x, z, p, m, q)
#    print(np.linalg.norm(Gproj-ffdata))
    return a, be, ce, de, s


def fdsid(fddata, n, q, estTrans=True, dtype='float', estimd=True, CT=False, T=1, W=np.empty(0)):
    """    
    Estimate a DT or CT state-space model from I/O frequency data
    
    Determines the (a,b,c,d,xt) parametrers such that (DT case)
     sum_i   || y[i,:] - d*u[i, :] + c*inv(z[i]*eye(n)-A)*[b,  xt]* [u[i, :]; z[i]]||^2_F
    is small where z = np.exp(1j*w)
    and CT Case
     sum_i   ||  y[i,:] - d*u[i, :] + c*inv(1j*w[i]*eye(n)-A)*b* u[i, :] ||^2_F
     
    
    Parameters
    ==========
    fddata : tuple
        fddata[0] = w 
            a vector of frequencies 
            in radians/sample
        fddata[1] = y 
            a matrix of the output transform data where y[i,:] is the DFT of the 
            time domain output signal corresponding to frequency w[i]
        fddata[2] = u 
            a matrix of the input transform data where u[i,:] is the DFT of the 
            time domain input signal corresponding to frequency w[i]
    n : integer
        the model order of the ss-model
    q : integer
        the numer of block rows used in the intermediate matrix. Must satisfy q>n
    estTrans : boolean, optional
        if True, a compensation for the transient term will be estimated (default)
    dtype : data type, optional
        if dtype = 'float' a real valued solution is returned.
        if dtype = 'complex' a complex valued solution is returned.
    estimd : boolean, optional
        if set to False no d matrix is esimated and a zero d matrix is returned
    CT : bool
        if True a CT model is estimated and estTrans is forced False.
    T :  a frequency scaling factor for the bilinear transformation used when CT=True. 
        Default is 1. If CT=False parameter T is disregarded 

        
    Returns
    =======
    a  : matrix_like
        the estimated a matrix
    b  : matrix_like
        the estimated b matrix
    c  : matrix_like
        the estimated c matrix
    d  : matrix_like
        the estimated d matrix
    xt : matrix_like
        vector of the transient compensation
    s  : matrix_like
        a vector of the singular values 
    """
    w = fddata[0]
    yd = fddata[1]
    ud = fddata[2]
    if CT:
        estTrans=False
        ad, bd, cd, dd, xt, s  = fdsid((cf2df(w,T), yd, ud), n, q, estTrans=False, dtype=dtype, estimd=True, CT=False, T=T, W=W)
        a, b, c, d = bilinear_d2c((ad, bd, cd, dd), T)
        if not estimd:
            b, d, resid = fdestim_bd(1j*w, yd, ud, a, c, estTrans, dtype, estimd, w=W)
            c, d, resid = fdestim_cd(1j*w, yd, ud, a, b, 0, estTrans, dtype, estimd, w=W)
            b, d, resid = fdestim_bd(1j*w, yd, ud, a, c, estTrans, dtype, estimd, w=W)
            c, d, resid = fdestim_cd(1j*w, yd, ud, a, b, 0, estTrans, dtype, estimd, w=W )
            xt = np.zeros((n, 1), dtype)
        return a, b, c, d, xt, s

    return gfdsid((np.exp(1j*w), yd, ud), n, q, estTrans=estTrans, dtype=dtype, estimd=estimd, w=W)


def gfdsid(fddata, n, q, estTrans=True, dtype='float', estimd=True, w=np.empty(0) ):
    """    
    Estimate a state-space model from I/O frequency data
    
    Determines the (a,b,c,d,xt) parametrers such that 
     sum_i || y[i,:] - d*u[i, :] + c*inv(z[i]*eye(n)-A)*[b,  xt]* [u[i, :]; z[i]] ||^2_w[i,:,:]
     is minimized 
    If estrTrans=False the following problem is solved
     sum_i ||y[i,:] - d*u[i, :] + c*inv(z[i]*eye(n)-A)*b * u[i, :] ||^2_w[i,:,:]
     is minimized 

    The weighted norm  || x ||^2_w is defined as  || w * x ||^2 where w is a square matrix such that w*w^H 
    is a positive definite matrix.  
    If the noise on y[i,:].T is a zero mean rv with covariance r[i,:,:] a BLUE estimator will be obtained if 
    w[i,:,:] is selected as linalg.cholesky(linalg.inv(r[i,:,:]])).T.conj()
    

    Parameters
    ==========
    fddata : tuple
        fddata[0] = z 
            a vector of complex scalars
        fddata[1] = y 
            a matrix of the output frequency data where y[i,:] corresponding to z[i]
        fddata[2] = u 
            a matrix of the input frrequency data where u[i,:] corresponding to z[i]
    n : integer
        the model order of the ss-model
    q : integer
        the numer of block rows used in the intermediate matrix. Must satisfy q>n
    estTrans : boolean, optional
        if True, a compensation for the transient term will be estimated (default)
    dtype : data type, optional
        if dtype = 'float' a real valued solution is returned.
        if dtype = 'complex' a complex valued solution is returned.
    estimd : boolean, optional
        if set to False no d matrix is esimated and a zero d matrix is returned
    w: matrix like, optional
        w[i,:,:] is the weighting matrix for data sample i, See above. 
    
    Returns
    =======
    a  : matrix_like
        the estimated a matrix
    b  : matrix_like
        the estimated b matrix
    c  : matrix_like
        the estimated c matrix
    d  : matrix_like
        the estimated d matrix
    xt : matrix_like
        vector of the transient compensation
    s  : matrix_like
        a vector of the singular values 
    """
    z = fddata[0]
    yd = fddata[1]
    ud = fddata[2]

    nwy, p = np.shape(yd)
    nwu, m = np.shape(ud)
    if estTrans:
        ude = np.concatenate((ud, np.asmatrix(z).T), 1)
        me = m + 1
    else:
        ude = ud
        me = m
    nw = np.size(z, 0)
    if nw != nwy:
        print("Error: z and Y sizes does not match!")
        return False
    if nw != nwu:
        print("Error: z and U sizes does not match!")
        return False
    y = np.empty([p*q, nw], dtype='complex')
    if estimd:
        u = np.empty([me*q, nw], dtype='complex')
    else:
        u = np.empty([me*(q-1), nw], dtype='complex')
    wf = np.ones((nw,1))
    if w.size>0:
        for widx in range(nw):
            wf[widx] = np.trace(w[widx,:,:])
    for widx in range(nw):
        y[:p, widx] = wf[widx]*yd[widx, :]
        u[:me, widx] = wf[widx]*ude[widx, :]
        zx = z[widx]
        for qidx in range(q)[1:]:
            y[qidx*p:(qidx+1)*p, widx] = wf[widx]*zx*yd[widx, :]
            if estimd or qidx<q-1:
                u[qidx*me:(qidx+1)*me, widx] = wf[widx]*zx*ude[widx, :]
            zx *= z[widx]
            
    if dtype == 'float':
        hu = np.concatenate((np.real(u), np.imag(u)), 1)
        hy = np.concatenate((np.real(y), np.imag(y)), 1)
    else:
        hu = u
        hy = y
    h = np.concatenate((hu, hy))
    r = linalg.qr(h.T, mode='r', overwrite_a=True)  #Calulates the projection
    r = r[0].T
    if estimd:
        r22 = r[me*q:, me*q:]
    else:
        r22 = r[me*(q-1):, me*(q-1):]                    
    
    u, s, vh = linalg.svd(r22, full_matrices=False, overwrite_a=True)
    c = u[:p, :n]
    lh = u[:p*(q-1), :n]
    rh = u[p:, :n]
    lsres = linalg.lstsq(lh, rh, overwrite_a=True, overwrite_b=True)
    a = lsres[0]
    if estTrans:
        b, d, xt, resid = fdestim_bd(z, yd, ud, a, c, estTrans, dtype, estimd, w)
        c, d, resid = fdestim_cd(z, yd, ud, a, b, xt, estTrans, dtype, estimd, w)
        b, d, xt, resid = fdestim_bd(z, yd, ud, a, c, estTrans, dtype, estimd, w)
        c, d, resid = fdestim_cd(z, yd, ud, a, b, xt, estTrans, dtype, estimd, w)
    else:   
        b, d, resid = fdestim_bd(z, yd, ud, a, c, estTrans, dtype, estimd, w)
        c, d, resid = fdestim_cd(z, yd, ud, a, b, 0, estTrans, dtype, estimd, w)
        b, d, resid = fdestim_bd(z, yd, ud, a, c, estTrans, dtype, estimd, w)
        c, d, resid = fdestim_cd(z, yd, ud, a, b, 0, estTrans, dtype, estimd, w)
        xt = np.zeros((n, 1), dtype)
    return a, b, c, d, xt, s


def ltifd_slow(a, b, u, z):
    """Calculates the (nz, n) size frequency kernel with input u
        fkern[:, i] = inv(eye(n)*z[i] - a)*b*u[i, :] 
        (slow version)
        """
    n = np.size(a, 0)
    nw = len(z)
    fkern = np.empty([n, nw], dtype='complex')
    for widx in range(nw):
        fkern[:, widx] = linalg.solve(np.eye(n) * z[widx] - a, np.dot(b, u[widx, :])) 
    return fkern

def ltifd(a, b, u, z, noWarning=False):
    """Calculates the nz x n frequency kernel with input u
        fkern[:, i] = inv(eye(n)*z[i] - a)*b*u[i, :] 
        (fast version)"""
    n = np.size(a, 0)
    nw = len(z)
    lam, t = linalg.eig(a)
    if np.linalg.matrix_rank(t) < n:
        if not noWarning:
            print("Matrix a defective. Eigenvectors do not form a basis. Using slow mode.")
        return ltifd_slow(a, b, u, z)
    else:
        it = linalg.inv(t)
        bb = np.dot(it, b)
        fkern = np.empty([n, nw], dtype='complex')
        for widx in range(nw):
            da = np.ones(n) * z[widx] - lam
            fkern[:, widx] = np.linalg.multi_dot([t, np.diag(1/da), np.dot(bb, u[widx, :])])    
        return fkern


def fdestim_cd(z, yd, ud, a, b, xt=0, estTrans=False, dtype='float', estimd=True, w = np.empty(0)):
    """Estimate c and d matrices given z, yd, ud and a, b, and (optionally xt) matrices
    
    Calulates the c and d matrices for a state-space representation of a rational function 
    given the a and b matrices and samples of the rational matrix function data in the form of 
    output vectors yd[i,:] and input vectors ud[i,:] corresponding to the rational function 
    evaluated at complex value z[i]. It solves 

    if estimd=True and estTrans=True    
    min_{c,d} sum_i || ([d 0] + c*inv(z[i]*I-a)*[b xt])*[ud[i,:]; z[i]]  - yd[i,:] |^2_w[i,:,:]

    if estimd=False and estTrans=True    
    min_{c} sum_i|| (c*inv(z[i]*I-a)*[b xt])*[ud[i,:]; z[i]]  - yd[i,:] |^2_w[i,:,:]

    if estimd=True and estTrans=False    
    min_{c,d} sum_i || (d+ c*inv(z[i]*I-a)*b)*ud[i,:]  - yd[i,:] |^2_w[i,:,:]

    if estimd=False and estTrans=False    
    min_{c} sum_i|| (c*inv(z[i]*I-a)*b)*ud[i,:]  - yd[i,:] |^2_w[i,:,:]
    
    The weighted norm  || x ||^2_w is defined as  || w * x ||^2 where w is a square matrix such that w*w^H 
    is a positive definite matrix.  
    If the noise on y[i,:].T is a zero mean rv with covariance r[i,:,:] a BLUE estimator will be obtained if 
    w[i,:,:] is selected as linalg.cholesky(linalg.inv(r[i,:,:]])).T.conj()


    if dtype='float' a real valued solution is calulated. if dtype='complex' 
    the solution is complex valued
    
    Parameters
    ----------
    yd : matrix_like 
        frequency data packed in a matrix. yd[i,:] is the output vector 
        corresponding to sample i
    ud : matrix_like 
        frequency data packed in a matrix. ud[i,:] is the output vector 
        corresponding to sample i
    z : matrix_like
        vector with the complex data z[i].
    a : matrix_like
        the a matrix
    b : matrix_like
        the b matrix
    dtype : optional
        data type of model either 'float' or 'complex'
    estimd : boolean, optional
        if set to False no d matrix is esimated and a zero d matrix is returned
      w: matrix like, optional
        w[i,:,:] is the weighting matrix for data sample i, See above. 
    
    Returns 
    -------
    
    c : matrix_like
        the LS-optimal c matrix
    d : matrix_like
        the LS-optimal d matrix

    """
    p = np.shape(yd)[1]
    m = np.shape(ud)[1]
    n = np.shape(a)[0]
    nz = np.size(z, 0)
    if estTrans == True:
        ude = np.concatenate((ud, np.array(z).reshape(nz, 1)), 1)
        be = np.concatenate((b, xt.reshape(n, 1)), 1)
    else:
        ude = ud
        be = b
    fkern = ltifd(a, be, ude, z)
    if estimd:
        r = np.empty((n+m, nz), dtype='complex')
        r[:n, :] = fkern
        r[n:, :] = np.transpose(ud)
    else:
        r = fkern            
    if w.size>0:
        nr = np.size(r,0)
        rw = np.empty((p*nz, nr*p), dtype='complex')
        ydw = np.empty((p*nz), dtype='complex')
        for zidx in range(nz): #Vectorize data and apply pre-whitening filter
            ydw[p*zidx:p*(zidx+1)] = np.matmul(w[zidx,:,:], yd[zidx,:])
            rw[p*zidx:p*(zidx+1),:] = np.kron(r[:,zidx].T, w[zidx,:,:])
            
        if dtype == 'float':
            lh = np.concatenate([np.real(ydw), np.imag(ydw)], 0)
            rh = np.concatenate([np.real(rw), np.imag(rw)], 0)
        else:
            lh = ydw
            rh = rw
        lsres = linalg.lstsq(rh, lh)
        vecCD = lsres[0]
        c = np.reshape(vecCD[:n*p], (p, n), order='F')
        if estimd:
            return c, np.reshape(vecCD[n*p:], (p, m), order='F'), lsres[1]
        else:
            return c , np.zeros((p, m), dtype), lsres[1] # Return c and d and residuals    
    else:
        if dtype == 'float':
            rh = np.concatenate([np.real(r), np.imag(r)], 1)
            lh = np.concatenate([np.real(yd), np.imag(yd)], 0)
        else:
            lh = yd
            rh = r
        lsres = linalg.lstsq(rh.T, lh)
        cd = lsres[0].T
        if estimd:
            return cd[:, :n], cd[:, n:], lsres[1] # Return c and d and residuals
        else:
            return cd, np.zeros((p, m), dtype), lsres[1] # Return c and d and residuals

        if estimd:
            return cd[:, :n], cd[:, n:], lsres[1] # Return c and d and residuals
        else:
            return cd, np.zeros((p, m), dtype), lsres[1] # Return c and d and residuals
        
def fdestim_bd(z, yd, ud, a, c, estTrans=False, dtype='float', estimd=True, w = np.empty(0)):
    """Estimate b and d (and optionally xt) matrices given z, yd, ud and a, c matrices
    
    Calulates the b and d (and optionally xt) matrices for a state-space representation of 
    a rational function 
    given the a and b matrices and samples of the rational matrix function data in the form of 
    output vectors yd[i,:] and input vectors ud[i,:] corresponding to the rational function 
    evaluated at complex value z[i]. It solves 
    
    if estimd=True and estTrans=True    
    min_{b,d,xt} sum_i || ([d 0] + c*inv(z[i]*I-a)*[b xt])[ud[i,:]; z[i]]  - yd[i,:] ||^2_w[i,:,:]

    if estimd=False and estTrans=True    
    min_{b,xt} sum_i|| (c*inv(z[i]*I-a)*[b xt])[ud[i,:]; z[i]]  - yd[i,:] ||^2_w[i,:,:]

    if estimd=True and estTrans=False    
    min_{b,d} sum_i || (d+ c*inv(z[i]*I-a)*b)ud[i,:]  - yd[i,:] ||^2_w[i,:,:]

    if estimd=False and estTrans=False    
    min_{b} sum_i|| (c*inv(z[i]*I-a)*b)ud[i,:]  - yd[i,:] ||^2_w[i,:,:]

    The weighted norm  || x ||^2_w is defined as  || w * x ||^2 where w is a square matrix such that w*w^H 
    is a positive definite matrix.  
    If the noise on y[i,:].T is zero mean rv with covariance r[i,:,:] a BLUE estimator will be obtained if 
    w[i,:,:] is selected as linalg.cholesky(linalg.inv(r[i,:,:]])).T.conj()

    if dtype='float' a real valued solution is calulated (default). if dtype='complex' 
    the solution is complex valued
    
    Parameters
    ----------
    z : matrix_like
        vector with the rational function evaluation points z[i].
    yd : matrix like
        output frequency data yd[i,:]    
    ud : matrix like
        input frequency data ud[i,:]    
    a : matrix_like
        the a matrix
    c : matrix_like
        the c matrix
    estTrans : boolean, optional
        if set to True also an xt vector will be estimated capturing the transient effect
    dtype : optional
        data type of model either 'float' or 'complex'
    estimd : boolean, optional
        if set to False no d matrix is esimated and a zero d matrix is returned
    w: matrix like, optional
        w[i,:,:] is the weighting matrix for data sample i, See above. 
    Returns 
    -------
    
    c : matrix_like
        the LS-optimal c matrix
    d : matrix_like
        the LS-optimal d matrix

    
    """
    p = np.shape(yd)[1]
    m = np.shape(ud)[1]
    n = np.shape(a)[0]
    nz = np.size(z, 0)
    if w.size>0:
        ydw = np.empty((nz,p),dtype='complex')
        for zidx in range(nz):
            ydw[zidx,:] = np.matmul(w[zidx,:,:], yd[zidx,:])
    else:
        ydw = yd
    if estTrans == True:
        ude = np.concatenate((ud, np.array(z).reshape(nz, 1)), 1)
 #       ude = np.concatenate((ud, np.ones((NwU, 1))), 1)
        me = m + 1
    else:
        ude = ud
        me = m

    fkern = ltifr(a.T, c.T, z).T
    if estimd:        
        r = np.empty([nz*p, me*n+m*p], dtype='complex') #room for B xt and D  
        if w.size>0:
            for zidx in range(nz):
                for midx in range(me):
                    r[p*zidx:p*(zidx+1), n*midx:n*(midx+1)] = np.matmul(
                            w[zidx,:,:], ude[zidx, midx] * fkern[:, :, zidx])
                for midx in range(m):
                    r[p*zidx:p*(zidx+1), me*n+p*midx:me*n+p*(midx+1)] = np.matmul(
                            w[zidx,:,:], ud[zidx, midx] * np.eye(p))
        else:
            for zidx in range(nz):
                for midx in range(me):
                    r[p*zidx:p*(zidx+1), n*midx:n*(midx+1)] = ude[zidx, midx] * fkern[:, :, zidx]
                for midx in range(m):
                    r[p*zidx:p*(zidx+1), me*n+p*midx:me*n+p*(midx+1)] = ud[zidx, midx] * np.eye(p)
    else:
        r = np.empty([nz*p, me*n], dtype='complex') #room for B xt     
        if w.size>0:
            for zidx in range(nz):
                for midx in range(me):
                    r[p*zidx:p*(zidx+1), n*midx:n*(midx+1)] = np.matmul(
                            w[zidx,:,:], ude[zidx, midx] * fkern[:, :, zidx])        
        else:
            for zidx in range(nz):
                for midx in range(me):
                    r[p*zidx:p*(zidx+1), n*midx:n*(midx+1)] = ude[zidx, midx] * fkern[:, :, zidx]        
    if dtype == 'float':
        rh = np.concatenate([np.real(r), np.imag(r)], 0)
        lh = np.concatenate([np.real(ydw.flatten()), np.imag(ydw.flatten())], 0)
    else:
        rh = r
        lh = ydw.flatten()
    lsres = linalg.lstsq(rh, lh)
    vecBD = lsres[0]
    vecB = vecBD[:n*me]
    if estimd:
        vecD = vecBD[n*me:]
    else:
        vecD = np.zeros(m*p)
    b = np.empty([n, m], dtype=dtype)
    d = np.empty([p, m], dtype=dtype)
    for midx in range(m):
        b[:, midx] = vecB[n*midx:n*(midx+1)]
        d[:, midx] = vecD[p*midx:p*(midx+1)]
    if estTrans:
        xt = vecB[n*m:n*(m+1)]
        return b, d, xt, lsres[1]
    else:
        return b, d, lsres[1]


def ffdata2fddata(ffdata, w):
    """Converts ffdata to fddata
    
    Converts frequency function data (ffdata) to input/output data format (fddata).

    Parameters
    ==========
    ffdata : array_like
        frequency function data in the format such that `ffdata[widx,:,:]` 
        corresponds to the frequency function matrix of size `(p,m)` at frequency index `widx` at a total number of frequencies `nw`
    w : array_like
        array with the corresponding frequencies in radians per sample.   
        
    Returns
    =======
    u : array_like
        Fourier transform of input of size `(m*nw,m)`
    y : array_like
        Fourier transform of output of size `(m*nw,p)`
    wn : array_like
        frequency vector of length `m*nw`
    """
    nw, p, m = np.shape(ffdata)
    u = np.empty([nw*m, m], dtype='complex')
    y = np.empty([nw*m, p], dtype='complex')
    wn = np.empty((nw*m))
    b0 = np.eye(m)
    for midx in range(m):
        y[midx*nw:(midx+1)*nw, :] = ffdata[:, :, midx]
        u[midx*nw:(midx+1)*nw, :] = np.tile(b0[midx, :], (nw, 1))
        wn[midx*nw:(midx+1)*nw] = w[:]
    return u, y, wn

def ltitr(a, b, u, x0=0, dtype='float'):
    """Calculates the time domain input to state respone
    
    Calculates the time domain state response
    x[i+1,:] = np.dot(a,x[i,:]) + np.dot(b,u[i,:])
    
    Parmeters
    =========
    a : array_like
        a square matrix of size `(n,n)`
    b : array_like
        a matrix of size `(n,m)`
    u : array_like
        an array of input vectors such that `u[i,:]` is 
        the input vector of size `m` at time index `i`. 
        The array has size `(N,m)`.
    x0 : array_like
        intial vector of size `n`, i.e. `x[0,:]=x0`. Default value is the zero vector. 
        
    Returns
    =======
    x : array_like
        the resulting state-sequence of size `(N,n)`
        x[k,:] is the state at sample k
    """
    n = np.shape(a)[0]
    N, m = np.shape(u)
    x = np.empty((N, n), dtype)
    if x0 == 0:
        x0 = np.zeros((n), dtype)
    x[0, :] = x0
    for nidx in range(N-1):
        x[nidx+1, :] = np.dot(a, x[nidx, :]) + np.dot(b, u[nidx, :])
    return x

def lsim(sys, u, x0=0, dtype='float'):
    """Calculates the time-domain output given input sequence and state-space model
    
    Calculates the time domain state response
    
    x[i+1,:] = np.dot(a,x[i,:]) + np.dot(b,u[i,:])
    
    y[i,:] = np.dot(c,x[i,:]) + np.dot(d,u[i,:])
    
    Parmeters
    =========
    sys : tuple
        sys = (a, b, c, d) or
        sys = (a, b, c)
        where
        a square matrix of size (n,n), b is 
        a matrix of size (n,m), c is a matrix of size (p,n)
        and (optionally) d is a matrix of size (p,m).
        
    u : array_like
        an array of input vectors such that u[i,:] is 
        the input vector of size m at time index i. 
        The array has size (N,m).
        
    x0 : array_like (optional)
        intial vector of size n, i.e. x[0,:]=x0. Default value is the zero vector. 
        
    Returns
    =======
    y : array_like
        the resulting output sequence of size (N,p)   """
        
    nn = len(sys)
    if nn == 3:
        a, b, c = sys
        p, nc = np.shape(c)
        nr, m = np.shape(b)
        d = np.zeros(p, m)
    elif nn == 4: 
        a, b, c, d = sys
        p, nc = np.shape(c)
        nr, m = np.shape(b)
    else:
        print("lsim: Incorrect number of matrices in sys.")
        return False
    x = ltitr(a, b, u, x0, dtype)
    nu = np.size(u, 0)
    y = np.empty((nu, p), dtype)
    for idx in range(nu):
        y[idx, :] = np.dot(c, x[idx, :]) + np.dot(d, u[idx, :])
    return y

def fdsim(sys, u, z, xt=np.empty(0)):
    """Calculates the output given input and state-space model in Fourier domain"""
    nwu, m = np.shape(u)
    nn = len(sys)
    if nn == 3:
        a, b, c = sys
        p, nc = np.shape(c)
        nr, m = np.shape(b)
        d = np.zeros(p, m)
    elif nn == 4: 
        a, b, c, d = sys
        p, nc = np.shape(c)
        nr, m = np.shape(b)
    else:
        print("fdsim: Incorrect number of matrices in sys.")
        return False
    y = np.empty((nwu, p), dtype='complex')
    if np.size(xt) > 0:
        ue = np.concatenate((u, z.reshape(nwu, 1)), 1)
        be = np.concatenate((b, xt.reshape(nr, 1)), 1)
        x = ltifd(a, be, ue, z)
    else:
        x = ltifd(a, b, u, z)
    for widx in range(nwu):
        y[widx, :] = np.dot(c, x[:, widx]) + np.dot(d, u[widx, :])
    return y

def bilinear_d2c(sys, T=1):
    """ Calculates the bilinear transformation D->C for ss-system sys """
    a, b, c, d = sys
    n =  np.shape(a)[0]
    ainv = np.linalg.inv(np.eye(n)+a)
    ac = np.dot(ainv, a-np.eye(n))*2/T
    bc = np.dot(ainv, b)*2/np.sqrt(T)
    cc = np.dot(c, ainv)*2/np.sqrt(T)
    dc = d - np.linalg.multi_dot([c, ainv, b]) 
    return ac, bc, cc, dc
    
def bilinear_c2d(sys, T=1):
    """ Calculates the bilinear transformation C->D for ss-system sys """
    a, b, c, d = sys
    n =  np.shape(a)[0]
    ainv = np.linalg.inv(np.eye(n)*2/T-a)
    ad = np.dot(np.eye(n)*2/T+a, ainv )    
    bd = np.dot(ainv, b)*2/np.sqrt(T)
    cd = np.dot(c, ainv)*2/np.sqrt(T)
    dd = d+np.linalg.multi_dot([c, ainv, b]) 
    return ad, bd, cd, dd

def cf2df(wc,T):
    """ Calculates the bilinear transformation frequency mapping C->D for frequency vector wc """
    return 2*np.arctan(wc*T/2)

def df2cf(wd,T):
    """ Calculates the bilinear transformation frequency mapping D->C for frequency vector wd """
    return 2*np.tan(wd/2)/T

def vec(a):
    """ Vectorize a matrix  
    
    Input
    =====
    a : matrix (ndarray)
    
    Returns
    =======
        A vector with all columns in a concatenated to one long vector
    """    
    return np.reshape(a,np.size(a), order='F')

def moebius(sys, par = (1, 0, 0, 1)):
    """ Calculates the bilinear transformation D->C for ss-system sys """
    a, b, c, d = sys
    alpha, beta, gamma, delta = par
    n =  np.shape(a)[0]
    ainv = np.linalg.inv(alpha*np.eye(n)-gamma*a)
    ac = np.dot(delta*a-beta*np.eye(n), ainv)
    bc = (alpha*delta-gamma*beta)*np.dot(ainv, b)
    cc = np.dot(c, ainv)
    dc = d + gamma*np.linalg.multi_dot([c, ainv, b]) 
    return ac, bc, cc, dc

def moebius_arg(z, par = (1, 0, 0, 1)):
    alpha, beta, gamma, delta = par
    nz = np.shape(z)[0]
    s = np.empty(nz, dtype=complex)
    for idx in np.arange(nz):
        s[idx]= (alpha*z[idx]+beta)/(gamma*z[idx]+delta)
    return s

def moebius_inv(sys, par = (1, 0, 0, 1)):
    """ Calculates the bilinear transformation D->C for ss-system sys """
    a, b, c, d = sys
    alpha, beta, gamma, delta = par
    n =  np.shape(a)[0]
    ainv = np.linalg.inv(delta*np.eye(n)+gamma*a)
    ac = np.dot((alpha*a +beta*np.eye(n)), ainv)
    bc = np.dot(ainv, b)
    cc = -(gamma*beta-alpha*delta)*np.dot(c, ainv)
    dc = d - gamma*np.linalg.multi_dot([c, ainv, b]) 
    return ac, bc, cc, dc
    alpha, beta, gamma, delta = par

def moebius_arg_inv(s, par = (1, 0, 0, 1)):
    alpha, beta, gamma, delta = par
    nz = np.shape(s)[0]
    z = np.empty(nz, dtype=complex)
    for idx in np.arange(nz):
        z[idx]= (beta-delta*s[idx])/(gamma*s[idx]-alpha)
    return z

def uq_cond(z, q):
    m = 1
    nw = np.size(z, 0)
    u = np.empty([m*q, nw*m], dtype='complex')
    for widx in range(nw):
        u[:m, widx*m:(widx+1)*m] = np.eye(m)
        zx = z[widx]
        for qidx in range(q)[1:]:
            u[qidx*m:(qidx+1)*m, widx*m:(widx+1)*m] = zx*np.eye(m)
            zx *= z[widx]
    return np.linalg.cond(u)

if __name__ == "__main__":

# Below is unit test code
    def unit_test_markov_kung():
        nmpset = [(4, 1, 1), (1, 1, 1), (4, 2, 1), (2, 4, 12)]
        for (n, m, p) in nmpset: 
            a = np.random.randn(n, n)
            b = np.random.randn(n, m)
            c = np.random.randn(p, n)
            mp = markov((a, b, c), 2*n )
            ae ,be, ce = kung_realization(mp, n)
            me = markov((ae, be, ce), 2*n )
            err = linalg.norm(me-mp)/linalg.norm(mp)
            if err > 1e-8:
                print('Unit test "markov_kung" failed', err, n, m, p)
                return False  
        print('Unit test "markov_kung" passed')
        return True

    def unit_test_ls_estim_cd():
        N = 100
        nmpset = [(4, 1, 1), (1, 1, 1), (2, 4, 12)]
        for (n, m, p) in nmpset: 
            A = np.random.randn(n, n)
            B = np.random.randn(n, m)
            C = np.random.randn(p, n)
            D = np.random.randn(p, m)
            fset = np.arange(0, N, dtype='float')/N
            z = np.exp(1j*2 * np.pi * fset)
            fd = fresp(z, A, B, C, D)
            Ce, De = ls_estim_cd(fd, z, A, B)
            fde = fresp(z, A, B, Ce, De)
            err = linalg.norm(fd-fde)/linalg.norm(fd)
            # print('|| H-He ||/||H|| = ', err)
            if err > 1e-8:
                print('Unit test "ls_estim_cd 1" failed')
                return False        
        for (n, m, p) in nmpset: 
            A = np.random.randn(n, n)
            B = np.random.randn(n, m)
            C = np.random.randn(p, n)
            D = np.zeros((p, m))
            fset = np.arange(0, N, dtype='float')/N
            z = np.exp(1j*2 * np.pi * fset)
            fd = fresp(z, A, B, C, D)
            Ce, De = ls_estim_cd(fd, z, A, B, estimd=False)
            fde = fresp(z, A, B, Ce, De)
            err = linalg.norm(fd-fde)/linalg.norm(fd)
            # print('|| H-He ||/||H|| = ', err)
            if err > 1e-8:
                print('Unit test "ls_estim_cd 2" failed')
                return False        
        for (n, m, p) in nmpset: 
            A = np.random.randn(n, n) + 1j*np.random.randn(n, n)
            B = np.random.randn(n, m) + 1j*np.random.randn(n, m)
            C = np.random.randn(p, n) + 1j*np.random.randn(p, n)
            D = np.random.randn(p, m) + 1j*np.random.randn(p, m)
            fset = np.arange(0, N, dtype='float')/N
            z = np.exp(1j*2 * np.pi * fset)
            fd = fresp(z, A, B, C, D)
            Ce, De = ls_estim_cd(fd, z, A, B, dtype='complex')
            fde = fresp(z, A, B, Ce, De)
            err = linalg.norm(fd-fde)/linalg.norm(fd)
            # print('|| H-He ||/||H|| = ', err)
            if err > 1e-8:
                print('Unit test "ls_estim_cd 3" failed')
                return False        
        print('Unit test "ls_estim_cd" passed')
        return True
    
    def unit_test_ls_estim_bd():
        N = 100
        nmpset = [(4, 1, 1), (1, 1, 1), (2, 4, 12)]
        for (n, m, p) in nmpset: 
            A = np.random.randn(n, n)
            B = np.random.randn(n, m)
            C = np.random.randn(p, n)
            D = np.random.randn(p, m)
            fset = np.arange(0, N, dtype='float')/N
            z = np.exp(1j*2 * np.pi * fset)
            fd = fresp(z, A, B, C, D)
            Be, De = ls_estim_bd(fd, z, A, C)
            fde = fresp(z, A, Be, C, De)
            err = linalg.norm(fd-fde)/linalg.norm(fd)
            # print('|| H-He ||/||H|| = ', err)
            if err > 1e-8:
                print('Unit test "ls_estim_bd" failed')
                return False        
        print('Unit test "ls_estim_bd" passed')
        return True
    
    def unit_test_transp_ffdata():
        N = 100
        nmpset = [(4, 1, 1), (1, 1, 1), (2, 4, 12)]
        for (n, m, p) in nmpset: 
            A = np.random.randn(n, n)
            B = np.random.randn(n, m)
            C = np.random.randn(p, n)
            D = np.random.randn(p, m)
            fset = np.arange(0, N, dtype='float')/N
            z = np.exp(1j*2 * np.pi * fset)
            fd = fresp(z, A, B, C, D)
            fde = transp_ffdata(transp_ffdata(fd))
            err = linalg.norm(fd-fde)/linalg.norm(fd)
            # print('|| H-He ||/||H|| = ', err)
            if err > 1e-8:
                print('Unit test "transp_fdata" failed')
                return False        
        print('Unit test "transp_fdata" passed')
        return True
    
    def unit_test_fresp():
        nw = 100
        nmpset = [(4, 1, 1), (1, 1, 1), (2, 4, 12)]
        for (n, m, p) in nmpset: 
            T = np.random.randn(n, n)
            s = linalg.svd(T, compute_uv=False)
            if s[n-1] < 1e-6:
                T = T + 1e-6*np.eye(n)
            B = np.random.randn(n, m)
            C = np.random.randn(p, n)
            D = np.random.randn(p, m)
            dd = np.zeros(n,dtype=complex)
            fset = np.arange(0, nw, dtype='float')/nw
            z = np.exp(1j*2 * np.pi * fset)
            adiag = np.random.randn(n)
            A = np.diag(adiag)
            fd = np.empty([nw, p, m], dtype='complex')
            for fidx in range(nw):   
                emad = z[fidx] * np.ones(n)
                for didx in range(n):
                    dd[didx] = 1/(emad[didx]-adiag[didx])  
                fd[fidx, :, :] = np.dot(np.dot(C, np.diag(dd)), B) + D
            Tinv = linalg.inv(T)
            Ae = np.dot(T, np.dot(A, Tinv))
            Be = np.dot(T, B)
            Ce = np.dot(C, Tinv)
            De = D
            fde = fresp_slow(z, Ae, Be, Ce, De)
            fdef = fresp(z, Ae, Be, Ce, De)
            err = linalg.norm(fd-fde)/linalg.norm(fd)
            errf = linalg.norm(fd-fdef)/linalg.norm(fd)
            # print('|| H-He ||/||H|| = ', err)
            if err > 1e-8:
                print('Unit test "fresp_slow" failed')
                return False        
            if errf > 1e-8:
                print('Unit test "fresp" failed', errf)
                return False        
        print('Unit test "fresp" "fresp_slow" passed')
        return True
    
    def unit_test_fresp_def():
        N = 100
        nmpset = [(2, 4, 12), (2, 3, 6)]
        for (n, m, p) in nmpset: 
            A = np.array([[0, 0], [1, 0]])
            B = np.random.randn(n, m)
            C = np.random.randn(p, n)
            D = np.random.randn(p, m)
            fset = np.arange(0, N, dtype='float')/N+1.0e-5
            z = np.exp(1j*2*np.pi*fset)
            frsp = fresp_slow(z, A, B, C, D)    
            frspf = fresp(z, A, B, C, D)
            err = linalg.norm(frspf-frsp)/linalg.norm(frsp)
            if err > 1e-8:
                print('Unit test "fresp_def" failed for a defective matrix', err)
                return False        
        print('Unit test "fresp_def" passed')
        return True
    
    
    def unit_test_fdestim_bd():
        N = 100
        nmpset = [(4, 1, 1), (1, 1, 1), (2, 4, 12)]
        for (n, m, p) in nmpset: 
            A = np.random.randn(n, n)
            B = np.random.randn(n, m)
            C = np.random.randn(p, n)
            D = np.random.randn(p, m)
            fset = np.arange(0, N, dtype='float')/N
            w = 2*np.pi*fset
            z = np.exp(1j*2*np.pi*fset)
            fd = fresp(z, A, B, C, D)
            U, Y, wn = ffdata2fddata(fd, w)
            zn = np.exp(1j*wn)
            Be, De, resid = fdestim_bd(zn, Y, U, A, C)
            fde = fresp(z, A, Be, C, De)
            err = linalg.norm(fd-fde)/linalg.norm(fd)
            # print('|| H-He ||/||H|| = ', err)
            if err > 1e-8:
                print('Unit test "fdestim_bd" failed')
                return False        
        print('Unit test "fdestim_bd" passed')
        return True

    def unit_test_fdestim_bd_w():
        N = 100 
        nmpset = [(4, 1, 1), (1, 1, 1), (2, 4, 12)]
        for (n, m, p) in nmpset: 
            A = np.random.randn(n, n)
            B = np.random.randn(n, m)
            C = np.random.randn(p, n)
            D = np.random.randn(p, m)
            W = np.empty((N*m,p,p));
            for nidx in range(N*m):
                W[nidx,:,:] = np.random.randn(p,p)
            fset = np.arange(0, N, dtype='float')/N
            w = 2*np.pi*fset
            z = np.exp(1j*2*np.pi*fset)
            fd = fresp(z, A, B, C, D)
            U, Y, wn = ffdata2fddata(fd, w)
            zn = np.exp(1j*wn)
            Be, De, resid = fdestim_bd(zn, Y, U, A, C, w=W)
            fde = fresp(z, A, Be, C, De)
            err = linalg.norm(fd-fde)/linalg.norm(fd)
            # print('|| H-He ||/||H|| = ', err)
            if err > 1e-8:
                print('Unit test "fdestim_bd_w" failed')
                return False        
            B = np.random.randn(n, m)+1j*np.random.randn(n, m)
            fd = fresp(z, A, B, C, D)
            U, Y, wn = ffdata2fddata(fd, w)
            zn = np.exp(1j*wn)
            Be, De, resid = fdestim_bd(zn, Y, U, A, C, dtype=complex, w=W)
            fde = fresp(z, A, Be, C, De)
            err = linalg.norm(fd-fde)/linalg.norm(fd)
            # print('|| H-He ||/||H|| = ', err)
            if err > 1e-8:
                print('Unit test "fdestim_bd_w" failed')
                return False        
        print('Unit test "fdestim_bd_w" passed')
        return True

    def unit_test_fdestim_bd_no_d():
        N = 100
        nmpset = [(4, 1, 1), (1, 1, 1), (2, 4, 12)]
        for (n, m, p) in nmpset: 
            A = np.random.randn(n, n)
            B = np.random.randn(n, m)
            C = np.random.randn(p, n)
            D = np.zeros((p, m))
            fset = np.arange(0, N, dtype='float')/N
            w = 2*np.pi*fset
            z = np.exp(1j*2*np.pi*fset)
            fd = fresp(z, A, B, C, D)
            U, Y, wn = ffdata2fddata(fd, w)
            zn = np.exp(1j*wn)
            Be, De, resid = fdestim_bd(zn, Y, U, A, C, estimd=False)
            fde = fresp(z, A, Be, C, De)
            err = linalg.norm(fd-fde)/linalg.norm(fd)
            # print('|| H-He ||/||H|| = ', err)
            if err > 1e-8:
                print('Unit test "fdestim_bd_no_d" failed')
                return False        
        print('Unit test "fdestim_bd_no_d" passed')
        return True

    
    def unit_test_fdestim_bd_cmplx():
        N = 100
        nmpset = [(4, 1, 1), (1, 1, 1), (2, 4, 12)]
        for (n, m, p) in nmpset: 
            A = np.random.randn(n, n)
            B = np.random.randn(n, m)
            B = B+1j*np.random.randn(n, m)
            C = np.random.randn(p, n)
            D = np.random.randn(p, m)
            D = D+ 1j*np.random.randn(p, m)
            fset = np.arange(0, N, dtype='float')/N
            w = 2*np.pi*fset
            z = np.exp(1j*2*np.pi*fset)
            fd = fresp(z, A, B, C, D)
            U, Y, wn = ffdata2fddata(fd, w)
            zn = np.exp(1j*wn)
            Be, De = fdestim_bd(zn, Y, U, A, C, dtype='complex')
            fde = fresp(z, A, Be, C, De)
            err = linalg.norm(fd-fde)/linalg.norm(fd)
            # print('|| H-He ||/||H|| = ', err)
            if err > 1e-8:
                print('Unit test "fdestim_bd cmplx" failed')
                return False        
        print('Unit test "fdestim_bd complex" passed')
        return True

    def unit_test_fdestim_cd():
        N = 100
        nmpset = [(4, 1, 1), (1, 1, 1), (2, 4, 12)]
        for (n, m, p) in nmpset:
            A = np.random.randn(n, n)
            B = np.random.randn(n, m)
            C = np.random.randn(p, n)
            D = np.random.randn(p, m)
            fset = np.arange(0, N, dtype='float')/N
            w = 2*np.pi*fset
            z = np.exp(1j*2*np.pi*fset)
            fd = fresp(z, A, B, C, D)
            U, Y, wn = ffdata2fddata(fd, w)
            zn = np.exp(1j*wn)
            Ce, De, resid = fdestim_cd(zn, Y, U, A, B)
            fde = fresp(z, A, B, Ce, De)
            err = linalg.norm(fd-fde)/linalg.norm(fd)
            # print('|| H-He ||/||H|| = ', err)
            if err > 1e-8:
                print('Unit test "fdestim_cd" failed')
                return False        
        print('Unit test "fdestim_cd" passed')
        return True

    def unit_test_fdestim_cd_no_d():
        N = 100
        nmpset = [(4, 1, 1), (1, 1, 1), (2, 4, 12)]
        for (n, m, p) in nmpset:
            A = np.random.randn(n, n)
            B = np.random.randn(n, m)
            C = np.random.randn(p, n)
            D = np.zeros((p,m))
            fset = np.arange(0, N, dtype='float')/N
            w = 2*np.pi*fset
            z = np.exp(1j*2*np.pi*fset)
            fd = fresp(z, A, B, C, D)
            U, Y, wn = ffdata2fddata(fd, w)
            zn = np.exp(1j*wn)
            Ce, De, resid = fdestim_cd(zn, Y, U, A, B, estimd=False)
            fde = fresp(z, A, B, Ce, De)
            err = linalg.norm(fd-fde)/linalg.norm(fd)
            # print('|| H-He ||/||H|| = ', err)
            if err > 1e-8:
                print('Unit test "fdestim_cd_no_d" failed')
                return False        
        print('Unit test "fdestim_cd_no_d" passed')
        return True
    
    

    def unit_test_fdestim_cd_w():
        N = 100
        nmpset = [(4, 1, 1), (1, 1, 1), (2, 4, 12)]
        for (n, m, p) in nmpset:
            A = np.random.randn(n, n)
            B = np.random.randn(n, m)
            C = np.random.randn(p, n)
            D = np.random.randn(p, m)
            W = np.empty((N*m,p,p));                        
            for nidx in range(N*m):
                W[nidx,:,:] = np.random.randn(p,p)
            fset = np.arange(0, N, dtype='float')/N
            w = 2*np.pi*fset
            z = np.exp(1j*2*np.pi*fset)
            fd = fresp(z, A, B, C, D)
            U, Y, wn = ffdata2fddata(fd, w)
            zn = np.exp(1j*wn)
            Ce, De, resid = fdestim_cd(zn, Y, U, A, B, w=W)
            fde = fresp(z, A, B, Ce, De)
            err = linalg.norm(fd-fde)/linalg.norm(fd)
            # print('|| H-He ||/||H|| = ', err)
            if err > 1e-8:
                print('Unit test "fdestim_cd_w" failed')
                return False        
            C = np.random.randn(p, n)+1j*np.random.randn(p, n)
            fd = fresp(z, A, B, C, D)
            U, Y, wn = ffdata2fddata(fd, w)
            zn = np.exp(1j*wn)
            Ce, De, resid = fdestim_cd(zn, Y, U, A, B, dtype='complex', w=W)
            fde = fresp(z, A, B, Ce, De)
            err = linalg.norm(fd-fde)/linalg.norm(fd)
            # print('|| H-He ||/||H|| = ', err)
            if err > 1e-8:
                print('Unit test "fdestim_cd_w" failed')
                return False        

        print('Unit test "fdestim_cd_w" passed')
        return True
    
    
    def unit_test_ltifr_slow():
        N = 100
        nw = N
        nmpset = [(4, 1, 1), (1, 1, 1), (2, 4, 12), (2, 3, 12)]
        for (n, m, p) in nmpset: 
            dd = np.zeros(n,dtype=complex)
            T = np.random.randn(n, n)
            s = linalg.svd(T, compute_uv=False)
            if s[n-1] < 1e-6:
                T = T + 1e-6*np.eye(n)
            B = np.random.randn(n, m)
            fset = np.arange(0, N, dtype='float')/N
            z = np.exp(1j*2*np.pi*fset)
            adiag = np.random.randn(n)
            A = np.diag(adiag)
            Tinv = linalg.inv(T)
            fkern1 = np.empty([nw, n, m], dtype='complex')
            for fidx in range(N):   
                emad = z[fidx]*np.ones(n)
                for didx in range(n):
                    dd[didx] = 1/(emad[didx]-adiag[didx])  
                fkern1[fidx, :, :] = np.dot(T, np.dot( np.diag(dd), B)) 
#
#
#                fkern1[fidx, :, :] = np.dot(T, np.dot(np.diag(
#                    map(lambda x: 1/x, emad-adiag)), B)) 
            Ae = np.dot(T, np.dot(A, Tinv))
            Be = np.dot(T, B)
            fkern = ltifr_slow(Ae, Be, z)
            fkernf = ltifr(Ae, Be, z)
            err = linalg.norm(fkern1-fkern)/linalg.norm(fkern1)
            errf = linalg.norm(fkern1-fkernf)/linalg.norm(fkern1)
            # print('|| H-He ||/||H|| = ', err)
            if err > 1e-8:
                print('Unit test "ltifr_slow" failed', err)
                return False        
            if errf > 1e-8:
                print('Unit test "ltifr" failed', errf)
                return False        
        print('Unit test "ltifr_slow" passed')
        print('Unit test "ltifr" passed')
        return True
    
    def unit_test_ltifr_def():
        N = 100
        nmpset = [(2, 4, 12), (2, 3, 6)]
        for (n, m, p) in nmpset:
            A = np.array([[0, 0], [1, 0]])
            B = np.random.randn(n, m)
            fset = np.arange(0, N, dtype='float')/N+1.0e-5
            z = np.exp(1j*2*np.pi*fset)
            fkern = ltifr_slow(A, B, z)    
            fkernf = ltifr(A, B, z)
            err = linalg.norm(fkernf-fkern)/linalg.norm(fkern)
            # print('|| H-He ||/||H|| = ', err)
            if err > 1e-8:
                print('Unit test "ltifr_def" failed for defective a matrix', err)
                return False        
        print('Unit test "ltifr_def" passed')
        return True
    
    def unit_test_bilinear():
        nmpset = [(4, 1, 1), (1, 1, 1), (2, 4, 12)]
        T = 2
        for (n, m, p) in nmpset: 
            A = np.random.randn(n, n)
            B = np.random.randn(n, m)
            C = np.random.randn(p, n)
            D = np.random.randn(p, m)
            a, b, c, d = bilinear_c2d((A, B, C, D), T)
            Ae, Be, Ce, De = bilinear_d2c((a, b, c, d), T)
            err = (linalg.norm(A-Ae)/linalg.norm(A) + 
                linalg.norm(B-Be)/linalg.norm(B) + 
                linalg.norm(C-Ce)/linalg.norm(C) +
                linalg.norm(D-De)/linalg.norm(D))
            if err > 1e-8:
                print('Unit test "bilinear 1" failed err=', err)
                return False        
            N = 100
            fset = np.arange(0, N, dtype='float')/N
            s = 1j*2*np.pi*fset
            z = np.exp(1j*cf2df(2*np.pi*fset,T))
            Hc = fresp(s, A, B, C, D)
            Hd = fresp(z, a, b, c, d)
            err = linalg.norm(Hc-Hd)/linalg.norm(Hc) 
            if err > 1e-8:
                print('Unit test "bilinear 2" failed err=', err)
                return False        
        print('Unit test "bilinear 1 and 2" passed')
        return True   

    def unit_test_fconv():
        N = 100
        T = 100
        fset = np.arange(-(N-1), N-1, dtype='float')
        wc = fset
        wd = cf2df(wc,T)
        w = df2cf(wd,T)
        err = linalg.norm(wc-w)/linalg.norm(wc) 
        if err > 1e-8:
            print('Unit test "fconv" failed')
            return False        
        print('Unit test "fconv" passed')
        return True   

    
    def unit_test_ffsid():
        N = 100
        nmpset = [(4, 1, 1), (1, 1, 1), (2, 4, 12)]
        for (n, m, p) in nmpset: 
            A = np.random.randn(n, n)
            B = np.random.randn(n, m)
            C = np.random.randn(p, n)
            D = np.random.randn(p, m)
            fset = np.arange(0, N, dtype='float')/N
            w = 2*np.pi*fset
            z = np.exp(1j*2*np.pi*fset)
            fd = fresp(z, A, B, C, D)
            Ae, Be, Ce, De, s =  ffsid(w, fd, n, n+1, dtype='float', estimd=True)
            fde = fresp(z, Ae, Be, Ce, De)
            err = linalg.norm(fd-fde)/linalg.norm(fd)
            if err > 1e-8:
                print('Unit test "ffsid" failed')
                return False        
        print('Unit test "ffsid" passed')
        return True
    def unit_test_ffsid_no_d():
        N = 100
        nmpset = [(4, 1, 1), (1, 1, 1), (2, 4, 12)]
        for (n, m, p) in nmpset: 
            A = np.random.randn(n, n)
            B = np.random.randn(n, m)
            C = np.random.randn(p, n)
          #  D = np.random.randn(p, m)
            D = np.zeros((p,m))
            fset = np.arange(0, N, dtype='float')/N
            w = 2*np.pi*fset
            z = np.exp(1j*2*np.pi*fset)
            fd = fresp(z, A, B, C, D)
            Ae, Be, Ce, De, s =  ffsid(w, fd, n, n+1, dtype='float', estimd=False)
            fde = fresp(z, Ae, Be, Ce, De)
            err = linalg.norm(fd-fde)/linalg.norm(fd)
            if err > 1e-8:
                print('Unit test "ffsid_no_d" failed')
                return False        
        print('Unit test "ffsid_no_d" passed')
        return True
    def unit_test_ffsid_complex():
        N = 100
        nmpset = [(4, 1, 1), (1, 1, 1), (2, 4, 12)]
        for (n, m, p) in nmpset: 
            A = np.random.randn(n, n)
            A = A + 1j*np.random.randn(n, n)
            B = np.random.randn(n, m)
            B = B+1j*np.random.randn(n, m)
            C = np.random.randn(p, n)
            D = np.random.randn(p, m)
            D = D+ 1j*np.random.randn(p, m)
            fset = np.arange(0, N, dtype='float')/N
            w = 2*np.pi*fset
            z = np.exp(1j*2*np.pi*fset)
            fd = fresp(z, A, B, C, D)
            Ae, Be, Ce, De, s =  ffsid(w, fd, n, n+1, dtype='complex', estimd=True)
            fde = fresp(z, Ae, Be, Ce, De)
            err = linalg.norm(fd-fde)/linalg.norm(fd)
            if err > 1e-8:
                print('Unit test "ffsid complex" failed')
                return False        
        print('Unit test "ffsid complex" passed')
        return True
    def unit_test_fdsid():
        N = 100
        nmpset = [(4, 1, 1), (1, 1, 1), (2, 4, 12)]
        for (n, m, p) in nmpset: 
            A = np.random.randn(n, n)
            lam = linalg.eig(A)[0]
            rho = np.max( np.abs(lam)) 
            ## Here we create a random stable DT system
            A = A/rho/1.01
            B = np.random.randn(n, m)
            C = np.random.randn(p, n)
            D = np.random.randn(p, m)
            fset = np.arange(0, N, dtype='float')/N
            w = 2*np.pi*fset
            z = np.exp(1j*2*np.pi*fset)
            fd = fresp(z, A, B, C, D)
            u = np.random.randn(N, m)
            y = lsim((A, B, C, D), u, dtype='float')
            yf = np.fft.fft(y,axis=0)
            uf = np.fft.fft(u,axis=0)
            fddata = (w, yf, uf)
            Ae, Be, Ce, De, xt, s =  fdsid(fddata, n, 2*n, 
                       estTrans=True, dtype='float')
            fde = fresp(z, Ae, Be, Ce, De)
            err = linalg.norm(fd-fde)/linalg.norm(fd)
            if err > 1e-8:
                print('Unit test "fdsid" failed')
                return False        
            W = np.random.randn(N,p,p)
#            for idx in range(N):
#                W[idx,:,:] = np.eye(p)
            Ae, Be, Ce, De, xt, s =  fdsid(fddata, n, 2*n, 
                                                estTrans=True, dtype='float', W=W)
            fde = fresp(z, Ae, Be, Ce, De)
            err = linalg.norm(fd-fde)/linalg.norm(fd)
            if err > 1e-8:
                print('Unit test "fdsid with weighting" failed')
                return False        
        print('Unit test "fdsid" passed')
        return True
    
    def unit_test_fdsid_noTrans():
        N = 100
        nmpset = [(4, 1, 1), (1, 1, 1), (2, 4, 12)]
        for (n, m, p) in nmpset: 
            A = np.random.randn(n, n)
            B = np.random.randn(n, m)
            C = np.random.randn(p, n)
            D = np.random.randn(p, m)
            fset = np.arange(0, N, dtype='float')/N
            w = 2*np.pi*fset
            z = np.exp(1j*2*np.pi*fset)
            fd = fresp(z, A, B, C, D)
            U, Y, wn = ffdata2fddata(fd, w)
            fddata = (wn, Y, U)
            Ae, Be, Ce, De, xt, s =  fdsid(fddata, n, 2*n, 
                       estTrans=False, dtype='float')
            fde = fresp(z, Ae, Be, Ce, De)
            err = linalg.norm(fd-fde)/linalg.norm(fd)
            if err > 1e-8:
                print('Unit test "fdsid_noTrans" failed')
                return False        
            W = np.random.randn(N*m,p,p)
#            for idx in range(N):
#                W[idx,:,:] = np.eye(p)
            Ae, Be, Ce, De, xt, s =  fdsid(fddata, n, 2*n, 
                                                estTrans=False, dtype='float', W=W)
            fde = fresp(z, Ae, Be, Ce, De)
            err = linalg.norm(fd-fde)/linalg.norm(fd)
            if err > 1e-8:
                print('Unit test "fdsid_notrans with weighting" failed')
                return False        
        print('Unit test "fdsid_notrans" passed')
        return True
            

    
    def unit_test_fdsid_no_d():
        N = 100
        nmpset = [(4, 1, 1), (1, 1, 1), (2, 4, 12)]
        for (n, m, p) in nmpset: 
            A = np.random.randn(n, n)
            lam = linalg.eig(A)[0]
            rho = np.max( np.abs(lam)) 
            ## Here we create a random stable DT system
            A = A/rho/1.01
            B = np.random.randn(n, m)
            C = np.random.randn(p, n)
            D = np.zeros((p, m))
            fset = np.arange(0, N, dtype='float')/N
            w = 2*np.pi*fset
            z = np.exp(1j*2*np.pi*fset)
            fd = fresp(z, A, B, C, D)
            u = np.random.randn(N, m)
            y = lsim((A, B, C, D), u, dtype='float')
            yf = np.fft.fft(y,axis=0)
            uf = np.fft.fft(u,axis=0)
            fddata = (w, yf, uf)
            Ae, Be, Ce, De, xt, s =  fdsid(fddata, n, 2*n, 
                       estTrans=True, dtype='float', estimd=False)
            fde = fresp(z, Ae, Be, Ce, De)
            err = linalg.norm(fd-fde)/linalg.norm(fd)
            if err > 1e-8:
                print('Unit test "fdsid_no_d" failed')
                return False        
        print('Unit test "fdsid_no_d" passed')
        return True

    def unit_test_fdsid_cmplx():
        N = 100
        nmpset = [(4, 1, 1), (1, 1, 1), (2, 4, 12)]
        for (n, m, p) in nmpset: 
            A = np.random.randn(n, n)
            lam = linalg.eig(A)[0]
            rho = np.max( np.abs(lam)) 
            ## Here we create a random stable DT system
            A = A/rho/1.01
            B = np.random.randn(n, m)
            B = B+1j*np.random.randn(n, m)
            C = np.random.randn(p, n)
            D = np.random.randn(p, m)
            D = D+ 1j*np.random.randn(p, m)
            fset = np.arange(0, N, dtype='float')/N
            w = 2*np.pi*fset
            z = np.exp(1j*2*np.pi*fset)
            fd = fresp(z, A, B, C, D)
            u = np.random.randn(N, m)
            y = lsim((A, B, C, D), u, dtype='complex')
            yf = np.fft.fft(y,axis=0)
            uf = np.fft.fft(u,axis=0)
            fddata = (w, yf, uf)
            Ae, Be, Ce, De, xt, s =  fdsid(fddata, n, 2*n, 
                                                estTrans=True, dtype='complex')
            fde = fresp(z, Ae, Be, Ce, De)
            err = linalg.norm(fd-fde)/linalg.norm(fd)
            if err > 1e-8:
                print('Unit test "fdsid_cmplx" failed')
                return False        
            W = np.random.randn(N,p,p)
#            for idx in range(N):
#                W[idx,:,:] = np.eye(p)
            Ae, Be, Ce, De, xt, s =  fdsid(fddata, n, 2*n, 
                                                estTrans=True, dtype='complex', W=W)
            fde = fresp(z, Ae, Be, Ce, De)
            err = linalg.norm(fd-fde)/linalg.norm(fd)
            if err > 1e-8:
                print('Unit test "fdsid_complex with weighting" failed')
                return False        
        print('Unit test "fdsid_complex" passed')
        return True
    
    def unit_test_lrm():
        N = 400
        nmpset = [(4, 1, 1), (1, 1, 1), (2, 4, 12)]
        for (n, m, p) in nmpset: 
            A = np.random.randn(n, n)
            lam = linalg.eig(A)[0]
            rho = np.max( np.abs(lam)) 
            ## Here we create a random stable DT system
            A = A/rho/1.01
            B = np.random.randn(n, m)
            C = np.random.randn(p, n)
            D = np.random.randn(p, m)
            fset = np.arange(0, N, dtype='float')/N
            z = np.exp(1j*2*np.pi*fset)
            fd = fresp(z, A, B, C, D)
            u = np.random.randn(N, m)
            y = lsim((A, B, C, D), u, dtype='float')
            ff = lrm(u,y)
            err = linalg.norm(fd-ff)/linalg.norm(fd)
            if err > 1e-3:
                print('Unit test "lrm 1" failed')
                return False                 
            ff = lrm(u,y, lam=1e-7)
            err = linalg.norm(fd-ff)/linalg.norm(fd)
            if err > 1e-3:
                print('Unit test "lrm 2" failed')
                return False     
            DFTind = np.arange(10,30,2)
            ff = lrm(u,y, DFTindices = DFTind, lam=1e-7)
            err = linalg.norm(fd[DFTind,: , :]-ff)/linalg.norm(fd[DFTind,: , :])
            if err > 1e-3:
                print('Unit test "lrm 3" failed')
                return False                 
        print('Unit test "lrm" passed')
        return True

    def unit_test_moebius():
        N = 100
        nmpset = [(4, 1, 1), (1, 1, 1), (2, 4, 12)]
        sset = np.random.randn(N)+1j*np.random.randn(N)
        par = np.random.randn(4)+1j*np.random.randn(4)
        zset = moebius_arg_inv(sset, par)
        sset1 = moebius_arg(zset, par)
        if linalg.norm(sset-sset1)/linalg.norm(sset)>1e-8:
            print('Unit test "moebius 0" failed')
            return False        
        for (n, m, p) in nmpset: 
            A = np.random.randn(n, n)
            B = np.random.randn(n, m)
            C = np.random.randn(p, n)
            D = np.random.randn(p, m)
            zset = np.random.randn(N)+1j*np.random.randn(N)
            par = np.random.randn(4)+1j*np.random.randn(4)
            sset = moebius_arg(zset, par)
            fd = fresp(sset, A, B, C, D)
            Ae, Be, Ce, De = moebius((A,B,C,D), par)
            fde = fresp(zset, Ae, Be, Ce, De)
            err = linalg.norm(fd-fde)/linalg.norm(fd)
            # print('|| H-He ||/||H|| = ', err)
            if err > 1e-8:
                print('Unit test "moebius 1" failed')
                print('err= ',err)
                return False        
            sset = np.random.randn(N)+1j*np.random.randn(N)
            zset = moebius_arg_inv(sset, par)
            fd = fresp(zset, A, B, C, D)
            Ae, Be, Ce, De = moebius_inv((A,B,C,D), par)
            fde = fresp(sset, Ae, Be, Ce, De)
            err = linalg.norm(fd-fde)/linalg.norm(fd)
            # print('|| H-He ||/||H|| = ', err)
            if err > 1e-8:
                print('Unit test "moebius 2" failed')
                print('err= ',err)
                return False        
            Ae, Be, Ce, De = moebius((A,B,C,D), par)
            a, b, c, d = moebius_inv((Ae,Be,Ce,De), par)
            sys = (a,b,c,d)
            sys0 = (A,B,C,D)
            err = 0
            for idx in np.arange(4):
                err += linalg.norm(sys[idx]-sys0[idx])/linalg.norm(sys0[idx])
            if err > 1e-8:
                print('Unit test "moebius 3" failed')
                print('err= ',err)
                return False        
                       
        print('Unit test "moebius" passed')
        return True


    
# Run the unit tests

    fsid()
    if (unit_test_fdestim_bd()
        and unit_test_fdestim_bd_no_d()
        and unit_test_fdestim_bd_w()
        and unit_test_fdestim_cd() 
        and unit_test_fdestim_cd_no_d() 
        and unit_test_fdestim_cd_w() 
        and unit_test_fresp() 
        and unit_test_fresp_def()
        and unit_test_ls_estim_bd() 
        and unit_test_ls_estim_cd() 
        and unit_test_transp_ffdata()
        and unit_test_fdsid()
        and unit_test_fdsid_noTrans()
        and unit_test_fdsid_no_d()
        and unit_test_fdsid_cmplx()
        and unit_test_ltifr_slow()
        and unit_test_ltifr_def()
        and unit_test_ffsid()
        and unit_test_ffsid_no_d()
        and unit_test_ffsid_complex()
        and unit_test_bilinear()
        and unit_test_fconv()
        and unit_test_lrm()
        and unit_test_moebius()
        and unit_test_markov_kung()):
        print("All unit tests passed")
        
