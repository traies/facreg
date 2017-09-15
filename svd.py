#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 10 20:18:28 2017

@author: traies
"""
import numpy as np
import random as rand
def two_norm(v):
    return np.sum(np.power(v, 2))

def two_by_two_eigval(M):
    a,b,c,d = M.flat
    T = a + d
    D = a * d - b * c
    square = np.sqrt(T ** 2 / 4 - D)
    l1 = T / 2 + square
    l2 = T / 2 - square
    return l1, l2


def tridiag(A):
    m = A.shape[0]
    n = A.shape[1]
    B = np.copy(A)
    b = False
    Q = np.eye(m)
#    if abs(B[1, 0]) < 10 ** -10:
#            B[1, 0] = 0
#            b = True
#    else:
    for k in range(n-2):
        v = np.zeros(n)
        alfa = - np.sign(B[k+1, k]) * np.linalg.norm(B[k+1:, k], ord=2)
        r = np.sqrt(0.5 * (alfa ** 2 - B[k+1, k] * alfa))
        v[k+1] = (B[k+1,k] - alfa) / (2 * r)
        v[k+2:] = B[k+2:, k] / (2 * r)
        Qk = np.eye(m) - 2 * np.outer(v,v)
        B = Qk.dot(B.dot(Qk))
        Q = np.dot(Q, Qk)
    
    return B, Q, b

def balance(A):
    n = A.shape[0]
    d = np.eye(n,n)
    B = np.array(A)
    for _ in range(2):
        for i in range(n):
            c = np.linalg.norm(B[:,i])
            r = np.linalg.norm(B[i,:])
            f = np.sqrt(r / c)
            d[i, i] = f * d[i,i]
            B[:, i] = f * B[:, i]
            B[i, :] = B[i, :] / f
                
    return B, d
                
def francis(A):
    B1, D = balance(A)
    Q = np.linalg.inv(D)
    
    B, Q1, _ = tridiag(B1)
    
    Q = np.dot(Q, Q1)
    
    i = 0
    j = B.shape[0] - 1
    
    aux1 = np.eye(3)
    nor = np.linalg.norm(A)
    while i < j - 1:
        p1, p2 = two_by_two_eigval(B[-2:, -2:])
        x = np.dot(B[i:i+3, i:i+2] - p1 * np.eye(3,2), B[i:i+2, i:i+1] - p2 * np.eye(2,1))
        
        u = x / np.linalg.norm(x)
        u = -u if u[0] < 0 else u
        u[0] += 1
        beta = 1 / u[0]
        p = aux1 - beta * np.dot(u,u.T)
        Q0 = np.eye(A.shape[0])
        Q0[i:i+3,i:i+3] = p
        Q = np.dot(Q, Q0)
        B = Q0.dot(B.dot(Q0))
        #check if bulge
#        if abs(B[i+2, i]) > 10 ** -10:
#            B[i:j+1, i:j+1], Qk, b = tridiag(B[i:j+1, i:j+1])
#            Q[i:j+1, i:j+1] = np.dot(Q[i:j+1, i:j+1], Qk)
#            i = i + 1 if b else i
            
        B[i:j+1, i:j+1], Qk, b = tridiag(B[i:j+1, i:j+1])
        Q[i:j+1, i:j+1] = np.dot(Q[i:j+1, i:j+1], Qk)
        i = i + 1 if b else i
#        else:
#            B, D = balance(B)
#            Q = np.dot(Q, np.linalg.inv(D))
            
        for x in range(i, j):
            if abs(B[x+1, x]) < 10 ** -10:
                B[x+1, x] = 0.0
                i += 1
            else:
                break
            
#        for x in range(j-1, i + 1, -1):
#            print(x, B[x+1, x])
#            if abs(B[x+1, x]) < 10 ** -10:
#                B[x+1, x] = 0.0
#                j -= 1
#            else:
#                break
        
#        if abs(B[j-2, j-1]) < np.finfo(np.float).eps * nor:
#            B[j, j-1] = 0
#            p1, p2 = two_by_two_eigval(B[-2:, -2:])
#            B[j-1, j-1] = p1
#            B[j,j] = p2
#            j -= 1
            
    if abs(B[j, j-1]) < np.finfo(np.float).eps * nor:
        B[j, j-1] = 0
    else:
        p1, p2 = two_by_two_eigval(B[-2:, -2:])
        B[j-1, j-1] = p1
        B[j,j] = p2
    return np.diagonal(B), Q

def house_red(A):
    m = A.shape[0]
    n = A.shape[1]
    B = np.copy(A)
    U = np.eye(m)
    V = np.eye(n)
    
#    Qk = np.eye(m)
    for k in range(n):
        print(k)
        v1 = np.array(B[k:,k])
        v1[0] += np.sign(v1[0]) * np.linalg.norm(v1, ord=2)
        v1 /= np.linalg.norm(v1, ord=2)
        
#        np.subtract(Qk[k:, k:], 2 * np.outer(v1, v1) / np.dot(v1, v1), Qk[k:, k:])
        np.subtract(B[k:, k:], 2 * np.outer(v1, np.dot(v1, B[k:, k:])) / np.dot(v1, v1), B[k:, k:])
#        B = np.dot(Qk, B)
#        U = np.dot(U, Qk)
#        if k < n-2:
#            a2 = B[k, k+1:]
#            v2 = np.array(a2)
#            v2[0] += np.sign(a2[0]) * np.linalg.norm(a2, ord=2 )
#            v2  /= np.linalg.norm(v2, ord=2)
#            v2 /= v2[0]
#            Pk = np.identity(n)
#            np.subtract(Pk[k+1:, k+1:], 2 * np.outer(v2, v2) / np.dot(v2, v2), Pk[k+1:, k+1:])
#            B = np.dot(B, Pk)
#            V = np.dot(Pk, V)

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
    A = np.matrix([[ 4., 2., 1.],
                   [ 3., 4., 2.],
                   [ 1., 2., 4.]])
#    
#    A = np.matrix([[  1.,   2.,   3.,],
#                 [  4.,   5.,   6.,],
#                 [  7.,   8.,   9.,],
#                 [ 10.,  -1.,  12.,]])
#    a1,_ = np.linalg.eig(A)
#    A = np.array([[1, 2, 3],
#                 [3, 4, 5],
#                 [6, 7, 8]])
    a = np.linalg.eig(A)[0]
    print(a)
    
    print(francis(A)[0])
    
#    B, Q = tridiag(A)
#    print(B)
#    
#    print(np.dot(Q, np.dot(B, Q.T)))
#    
#    a2,_ = np.linalg.eig(B)
#    print(a1, '\n', a2)
      
#    gr_svd(A)
#    print(B)
#    print(V)
#    print(U * B * V.T)
#    print(B, '\n')
#    print(U * B * V.T)