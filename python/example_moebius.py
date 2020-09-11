#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 10 13:17:07 2020

@author: mckelvey
"""
import numpy as np
from scipy import linalg
import fsid 


n = 2
m = 3
p = 4
N = 100
q = 20
T = 1
sigma = 0.001

A = np.random.randn(n, n)
B = np.random.randn(n, m)
C = np.random.randn(p, n)
D = np.random.randn(p, m)
D = D*0
# Create frequency function (Laplace domain)
fset = np.arange(0, N, dtype='float')/N
w = 2*np.pi * fset
#    wexp = np.exp(1j*2 * np.pi * fset)
sset = 1j*w
fd = fsid.fresp(sset, A, B, C, D) +sigma*(np.random.randn(N,p,m) + 1j*np.random.randn(N,p,m))

# Use the Möbius transformation corresponding to the bilinear transformation
par =(2, -2, T, T)  
wd = fsid.cf2df(w,T)
zset = fsid.moebius_arg_inv(sset, (2,-2,T,T))

# estimate using non-transformed formulation
Ae1, Be1, Ce1, De1, s = fsid.gffsid(sset, fd, n, q)

# Estimate ss model from ffdata 
Ae, Be, Ce, De, s = fsid.gffsid(zset, fd, n, q)


# Convert to CT
Aec, Bec, Cec, Dec =  fsid.moebius_inv((Ae, Be, Ce, De), par)
# Frequency response of estimated model
fde = fsid.fresp(sset, Aec, Bec, Cec, Dec)
fde1 = fsid.fresp(sset, Ae1, Be1, Ce1, De1)
print('With transf CT FF: || G-Ge ||/||G|| = ', linalg.norm((fd-fde)/linalg.norm(fd)))
print('without transf CT FF: || G-Ge ||/||G|| = ', linalg.norm((fd-fde1)/linalg.norm(fd)))
print('After trans: Condition number for Uq = ', '{0:1.2e}'.format(fsid.uq_cond(zset, q)))
print('Before trans: Condition number for Uq = ', '{0:1.2e}'.format(fsid.uq_cond(sset, q)))

## Cehck if DFT grid is good?
zset_dft = np.exp(1j*2*np.pi*np.arange(N, dtype='float')/N)
print('DFT: Condition number for Uq = ', '{0:1.2e}'.format(fsid.uq_cond(zset_dft, q)))
zset_dft = np.exp(1j*2*np.pi*np.arange(N/2, dtype='float')/N)
print('Half DFT: Condition number for Uq = ', '{0:1.2e}'.format(fsid.uq_cond(zset_dft, q)))
zset_dft = np.exp(1j*2*np.pi*np.arange(N/4, dtype='float')/N)
print('Quarter DFT: Condition number for Uq = ', '{0:1.2e}'.format(fsid.uq_cond(zset_dft, q)))

 
#Ae, Be, Ce, De, s = fsid.ffsid(w, fd, n, n*2, CT=True, T=T, estimd=False)
#fde2 = fsid.fresp(1j*w, Ae, Be, Ce, De)
#print('CT2 FF: || G-Ge ||/||G|| = ', linalg.norm(fd-fde2)/linalg.norm(fd))
#print(De)
