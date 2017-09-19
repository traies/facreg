#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 10 20:18:28 2017

@author: traies
"""
import numpy as np

epsilon = 10 ** -5

def two_by_two_eigval(M):
    a,b,c,d = M.flat
    T = a + d
    D = a * d - b * c
    square = np.sqrt(T ** 2 / 4 - D)
    l1 = T / 2 + square
    l2 = T / 2 - square
    return l1, l2


def tridiag(B):
    m = B.shape[0]
    n = B.shape[1]
    Q = np.eye(m)
    for k in range(n-1):
        v = np.zeros(n)
        alfa = - np.sign(B[k+1, k]) * np.linalg.norm(B[k+1:, k], ord=2)
        r = np.sqrt(0.5 * (alfa ** 2 - B[k+1, k] * alfa))
        if abs(r) < epsilon:
            break
        v[k+1] = (B[k+1,k] - alfa) / (2 * r)
        v[k+2:] = B[k+2:, k] / (2 * r)
        Qk = np.eye(m) - 2 * np.outer(v,v)
        B = Qk @ B @ Qk
        Q = Q @ Qk
    return B, Q

def chase(B):
    B = np.copy(B)
    m = B.shape[0]
    n = B.shape[1]
    Q = np.eye(m)
    v = np.zeros(3)
    for k in range(n-1):
        
        alfa = - np.sign(B[k+1, k]) * np.linalg.norm(B[k+1:k+4, k], ord=2)
        r = np.sqrt(0.5 * (alfa ** 2 - B[k+1, k] * alfa))
        if abs(r) < epsilon:
            break
        v[0] = (B[k+1,k] - alfa) / (2 * r)
        Qk = np.eye(n)
        if k < n - 3:
            v[1:] = B[k+2:k+4, k] / (2 * r)
            Qk[k+1:k+4, k+1:k+4] = Qk[k+1:k+4, k+1:k+4] - 2 * np.outer(v,v)
        elif k < n - 2:
            v[1:n-k-1] = B[k+2:n, k] / (2 * r)
            Qk[k+1:n, k+1:n] = Qk[k+1:n, k+1:n] - 2 * np.outer(v[0:n-k-1],v[0:n-k-1])
        else:
            Qk[k+1, k+1] = Qk[k+1, k+1] - 2 * np.outer(v[0],v[0])
        B = Qk @ B @ Qk
        Q = Q @ Qk
    
    return Q
                
def sort_eigval(B, Q):
    sortlist = np.argsort(B)
    Bp = np.empty(B.shape)
    Qp = np.empty(Q.shape)
    j = 0
    for i in reversed(sortlist):
        Bp[j] = B[i]
        Qp[:, j] = Q[:, i]
        j += 1
    return Bp, Qp
    
def francis(A):
    B = np.copy(np.asarray(A))
    B, Q = tridiag(B)
    
    i = 0
    j = 0
    
    aux1 = np.eye(3)
    while i < B.shape[0] - 2:
        if  i >= j:
            i = j
            j = B.shape[0] - 1
        
        while i < j and i < B.shape[0] - 2:
            if j - i >= 2:
                p1, p2 = two_by_two_eigval(B[j-1:j+1, j-1:j+1])
            else:
                p1, p2 = two_by_two_eigval(B[-2:,-2:])
            
            x = np.dot(B[i:i+3, i:i+2] - p1 * np.eye(3,2), B[i:i+2, i:i+1] - p2 * np.eye(2,1))
            u = x / np.linalg.norm(x)
            u = -u if u[0] < 0 else u
            u[0] += 1
            beta = 1 / u[0]
            p = aux1 - beta * np.dot(u,u.T)
            Q0 = np.eye(A.shape[0])
            Q0[i:i+3,i:i+3] = p
            Q = Q @ Q0
            
            B = Q0 @ B @ Q0
            
            # check if bulge
            if abs(B[i+2, i]) > epsilon:
                Qk = chase(B[i:j+1, i:j+1])
                Q0 = np.eye(A.shape[0])
                Q0[i:j+1, i:j+1] = Qk
                Q = Q @ Q0
                B = Q0.T @ B @ Q0
           
            for x in range(i, j):
                if abs(B[x+1, x]) < epsilon:
                    B[x+1, x] = 0.0
                    i += 1
                else:
                    break
                
            for x in range(i, j):
                if abs(B[x+1, x]) < epsilon:
                    B[x+1, x] = 0.0
                    j = x
                    break
        i += 1
    B = np.copy(np.diagonal(B))
    B, Q = sort_eigval(B, Q)
    return B, Q
    
    
if __name__ == "__main__":
#    A = np.matrix([[ 4., 1., -2., 2.],
#                   [ 1., -2.,  0., 1.],
#                   [-2., 0.,  3., -2.],
#                   [ 2., 1., -2., -1.]])
    A = np.matrix([[ 4., 3., 1.],
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
    
    a, e = np.linalg.eig(A)
    print(a, '\n', e)
    print(np.dot(e, np.dot(np.diag(a), e.T)))
    
    a1, e1 = francis(A)
#    e1[:,0] = - e1[:,0]
#    aux = np.copy(e1[:, 2])
#    e1[:, 2] = e1[:,1]
#    e1[:, 1] = aux
#    aux1 = a1[2]
#    a1[2] = a1[1]
#    a1[1] = aux1
    print(a1, '\n', e1)
    print("\n")
    print(np.dot(e1, np.dot(np.diag(a1), e1.T)))
    
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