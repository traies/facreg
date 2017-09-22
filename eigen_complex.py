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
    Q = np.eye(m, dtype=complex)
    for k in range(n-1):
        v = np.zeros(n, dtype=complex)
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
    Q = np.eye(m, dtype=complex)
    v = np.zeros(3, dtype=complex)
    for k in range(n-1):
        
        alfa = - np.sign(B[k+1, k]) * np.linalg.norm(B[k+1:k+4, k], ord=2)
        r = np.sqrt(0.5 * (alfa ** 2 - B[k+1, k] * alfa))
        if abs(r) < epsilon:
            break
        v[0] = (B[k+1,k] - alfa) / (2 * r)
        Qk = np.eye(n, dtype=complex)
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
    Bp = np.empty(B.shape, dtype=complex)
    Qp = np.empty(Q.shape, dtype=complex)
    j = 0
    for i in reversed(sortlist):
        Bp[j] = B[i]
        Qp[:, j] = Q[:, i]
        j += 1
    return Bp, Qp

def get_eigvect(T):
    n = T.shape[0]
    
    eigvect = np.zeros([n, n], dtype=complex)
    
    for i in reversed(range(n)):
        eigvect[:i, i] = - np.linalg.inv(T[:i, :i] - T[i,i] * np.eye(i, i)) @ T[:i, i]
        eigvect[i, i] = 1
        eigvect[i+1:, i] = 0
    
    eigval = np.diagonal(T)
    return eigval, eigvect

def francis(A):
    B = np.copy(np.asarray(A))
    B, Q = tridiag(B)
    i = 0
    j = 0
    
    aux1 = np.eye(3, dtype=complex)
    while i < B.shape[0] - 2:
        if  i >= j:
            i = j
            j = B.shape[0] - 1
        
        while i < j and i < B.shape[0] - 2:
            if j - i >= 2:
                p1, p2 = two_by_two_eigval(B[j-1:j+1, j-1:j+1])
            else:
                p1, p2 = two_by_two_eigval(B[-2:,-2:])
            
#            print(p1, p2)
            x = np.asarray((B[i:i+3, i:i+2] - p1 * np.eye(3,2)) @ (B[i:i+2, i:i+1] - p2 * np.eye(2,1)))
            x = np.real_if_close(x)
            u = x / np.linalg.norm(x)
            u = -u if u[0] < 0 else u
            u[0] += 1
            beta = 1 / u[0]
            
            p = aux1 - beta * np.dot(u,u.T)
            Q0 = np.eye(A.shape[0], dtype=complex)
            Q0[i:i+3,i:i+3] = p
            
            Q = Q @ Q0
            B = Q0 @ B @ Q0
            
            # check if bulge
            if i < B.shape[0] - 2 and abs(B[i+2, i]) > epsilon:
                Qk = chase(B[i:j+1, i:j+1])
                Q0 = np.eye(A.shape[0], dtype=complex)
                Q0[i:j+1, i:j+1] = Qk
                Q = Q @ Q0
                B = Q0.T @ B @ Q0
                
            for x in range(i, j):
                print(x, B[x+1, x])
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
    
    n = B.shape[0]
    A = B[n-2:, n-2:]
    while abs(A[1, 0]) > epsilon:
      # Gram schmidt qr decomp.
      a1 = A[:, 0]
      a2 = A[:, 1]
      u1 = a1
      e1 = u1 / np.linalg.norm(u1)
      u2 = a2 - ((u1.conj().T @ a2) / (u1.conj().T @ u1)) * u1
      e2 = u2 / np.linalg.norm(u2)
      Q1 = np.array([e1, e2]).T
      R = np.zeros([2,2], dtype=complex)
      R[0,0] = e1.conj().T @ a1
      R[0,1] = e1.conj().T @ a2
      R[1,1] = e2.conj().T @ a2
      Q0 = np.eye(n, n, dtype=complex)
      Q0[n-2:,n-2:] = Q1
      Q = Q @ Q0
      A = R @ Q1
      B = Q0.T @ B @ Q0
    
    
    D = np.copy(np.diagonal(B))
    eigval, eigvect = get_eigvect(B)
    Q = Q @ eigvect
    
    D, Q = sort_eigval(eigval, Q)
    return D, Q
    
    
if __name__ == "__main__":
#    A = np.matrix([[ 4., 1., -2., 2.],
#                   [ -1., -2.,  0., 1.],
#                   [-2., 0.,  3., -2.],
#                   [ 2., 1., -2., -1.]])
    A = np.matrix([[0 , 0., 1.],
                   [ 0., 1., 0.],
                   [ -1., 0., 0.]])
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
    print(np.dot(e, np.dot(np.diag(a), np.linalg.inv(e))))
    
#    a1, e1 = francis(A)
#    e1[:,0] = - e1[:,0]
#    aux = np.copy(e1[:, 2])
#    e1[:, 2] = e1[:,1]
#    e1[:, 1] = aux
#    aux1 = a1[2]
#    a1[2] = a1[1]
##    a1[1] = aux1
#    print('\n')
#    print(a1, '\n', e1)
#    print(np.dot(e1, np.dot(np.diag(a1), np.linalg.inv(e1))))
    
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
