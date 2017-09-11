#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 10 20:18:28 2017

@author: traies
"""
import numpy as np

def two_norm(v):
    return np.sum(np.power(v, 2))

def house_red(A):
    m = A.shape[0]
    n = A.shape[1]
    B = np.matrix(A)
    U = np.eye(m)
    V = np.eye(n)
    print(A)
    for k in range(n):
        a = B[k:,k]
        v1 = np.matrix(a)
        v1[0] += np.sign(a[0]) * np.linalg.norm(a, ord=2)
        v1 /= np.linalg.norm(v1, ord=2)
        v1 /= v1[0]
        Qk = np.identity(m)
        Qk[k:, k:] -= 2 * v1 * v1.T / (v1.T * v1)
        
        B = np.dot(Qk, B)
        print(k)
        U = np.dot(U, Qk)
        if k < n-2:
            a2 = np.array(B[k, k+1:])
            v2 = np.matrix(a2)
            v2[0, 0] += np.sign(a2[0, 0]) * np.linalg.norm(a2, ord=2, axis=1)
            v2[0]  /= np.linalg.norm(v2, ord=2, axis=1)
            v2[0] = v2[0] / v2[0, 0]
            Pk = np.identity(n)
            Pk[k+1:, k+1:] -=  2 * (v2.T * v2) / (v2 * v2.T)
            B = B * Pk
            V = np.dot(Pk, V)
            
    print(A)
    return B, U, V

def demm_kah_rot(f, g):
    if abs(f) <  np.finfo(np.float).eps:
        c, s, r = 0, 1, g
        return c, s, r
    elif abs(f) > abs(g):
        t = g / f
        t1 = np.sqrt(1 + np.power(t, 2))
        c = 1 / t1
        s = t * c
        r = f * t1
        return c, s, r
    else:
        t = f / g
        t1 = np.sqrt(1 + np.power(t, 2))
        s = 1 / t1
        c = t * s
        r = g * t1
        return c, s, r

def sweep(B):
    n = B.shape[1]
    d = np.array(np.diagonal(B))
    e = np.array(np.diagonal(B, offset=1))
    
    u, s, v = np.linalg.svd(B)
    Rp = np.eye(n)
    Rq = np.eye(n)
    while np.any(np.abs(e) > np.finfo(np.float).eps * 100):
        c = 1
        co = 1
        so = 1
        for i in range(0, n-1):
            if abs(e[i]) <= np.finfo(np.float).eps * 100:
                e[i] = 0
        for i in range(0, n-1):
            c, s, r = demm_kah_rot(c * d[i], e[i])
            if i != 0:
                e[i-1] = r * so
            Rpa = np.eye(n)
            Rpa[i:i+2, i:i+2] = np.mat([[c, s], [-s, c]])
            
            Rp = np.dot(Rp, Rpa)
            co, so, d[i] = demm_kah_rot(co * r, d[i+1] * s)
            
            Rqa = np.eye(n)
            Rqa[i:i+2, i:i+2] = np.mat([[co, so], [-so, co]])
            Rq = np.dot(Rqa, Rq)
        h = c * d[n-1]
        e[n-2] = h * so
        d[n-1] = h * co
    B[:n, :n] = np.diag(np.abs(d))
    return np.abs(d), B

def gr_svd(A):
    B, U, V = house_red(A)
    
    for i in range(B.shape[0]):
        for j in range(B.shape[1]):
            if abs(B[i,j]) <= np.finfo(np.float32).eps :
                B[i, j] = 0
    
    b, B = sweep(B)
    return b
        
    
    
if __name__ == "__main__":
#    A = np.matrix([[ 4., 1., -2., 2.],
#                   [ 1., -2.,  0., 1.],
#                   [-2., 0.,  3., -2.],
#                   [ 2., 1., -2., -1.]])
#    A = np.matrix([[ 4., 2., 1.],
#                   [ 3., 4., 2.],
#                   [ 1., 2., 4.]])
#    
    A = np.matrix([[  1.,   2.,   3.,],
                 [  4.,   5.,   6.,],
                 [  7.,   8.,   9.,],
                 [ 10.,  -1.,  12.,]])
               
    print(gr_svd(A))
    u, s, v = np.linalg.svd(A)
    print(s)
      
#    gr_svd(A)
#    print(B)
#    print(V)
#    print(U * B * V.T)
#    print(B, '\n')
#    print(U * B * V.T)