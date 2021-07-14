#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 17 11:10:38 2018

@author: mckelvey
"""
import numpy as np
from scipy import linalg
import fsid 
import time

#import matplotlib.pyplot as plt

Nset = [100, 200, 400, 800]
nset= [2, 5, 10, 20]
m = 2;
p = 8;
MC = 100;

tim_res = np.zeros((4,4))
nidx = 0;
for n in nset:
    Nidx = 0
    for N in Nset:
        A = np.random.randn(n, n)
        lam = linalg.eig(A)[0]
        rho = np.max( np.abs(lam)) 
        ## Here we create a random stable DT system
        A = A/rho/1.01
        B = np.random.randn(n, m)
        C = np.random.randn(p, n)
        D = np.random.randn(p, m)
        fset = np.arange(0, N, dtype='float')/N
        u = np.random.randn(N, m)
        y = fsid.lsim((A, B, C, D), u, dtype='float')
    
        yf = np.fft.fft(y,axis=0)
        uf = np.fft.fft(u,axis=0)
        fset = np.arange(0, N, dtype='float')/N
        wexp = np.exp(1j*2 * np.pi * fset)
        W = np.zeros((N, p, p))
        for i in range(N):
            W[i,:,:] = np.eye(p)  
        tic =  time.mktime(time.gmtime())
        for mc in range(MC):
#            Ae, Be, Ce, De, xt, s =  fsid.gfdsid((wexp, yf, uf), n, n+1, estTrans=True, estimd=True, w=W)
             Ae, Be, Ce, De, xt, s =  fsid.gfdsid((wexp, yf, uf), n, n+1, estTrans=True, estimd=True)
        tim_res[Nidx,nidx] = time.mktime(time.gmtime())-tic
        fde = fsid.fresp(wexp, Ae, Be, Ce, De)
        fd = fsid.fresp(wexp, A, B, C, D)        
        print('|| H-He ||/||H|| = ', linalg.norm(fd-fde)/linalg.norm(fd))
        Nidx = Nidx + 1
    nidx = nidx + 1
tim_res = tim_res/MC; #timeit.timeit() has unit 1000 seconds

print tim_res
