#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  6 14:42:01 2017

@author: traies
"""

import numpy as np

def save_8_bit_pgm(filepath, arr, width, height):
    with open(filepath, 'wb') as img:
        img.write(b'P5\n')
        img.write(b''+bytes(str(width), "ASCII")+b' ')
        img.write(b''+bytes(str(height), "ASCII")+b'\n')
        img.write(b'255\n')
        for i in range(len(arr)):
            img.write(bytes([arr[i]]))

def load_8_bit_pgm(filepath): 
    with open(filepath, 'rb') as img:
        img.readline() # Magic number, b"P5"
        width, height = [int(x) for x in img.readline().split()]
        img.readline() # maxgrey, should be 255
        arr = np.empty(width*height)
        for i in range(height*width):
            arr[i] = int(ord(img.read(1)))
    return arr

def closest_aprox_eucl_distance(proj_vect, train_mat):
    a = -1
    mi = np.Infinity
    for i in range(train_mat.shape[0]):
        m = np.sqrt(np.sum(np.power(train_mat[i,:] - proj_vect, 2)))
        if m <= mi:
            mi = m
            a = i
    return a

def get_kernel_accuracy(eigvect_k, train_mat, mean, og_mat, base_path, subjects, samples, base_samples, exp):
    
    # get averages matrix
    avg_mat = []
    for i in range(subjects):
        subj_list = []
        for j in range(base_samples):
            subj = np.matrix(load_8_bit_pgm(base_path + "/s"+str(i+1)+"/"+str(j+1)+".pgm"))
            subj -= mean
            proj = np.power(subj * og_mat.T, exp) * eigvect_k
            subj_list.append(np.squeeze(np.asarray(proj)))
        subj_list_m = np.matrix(subj_list)
        x = [np.mean(subj_list_m[:, i]) for i in range(subj_list_m.shape[1])]
        avg_mat.append(x)
    avg_mat = np.matrix(avg_mat)
    
    success  = 0
    fail = 0
    for i in range(subjects):
        for j in range(base_samples, samples):
            test = np.matrix(load_8_bit_pgm(base_path + "/s"+str(i+1)+"/"+str(j+1)+".pgm"))
            test -= mean
            proj = np.power(test * og_mat.T, exp) * eigvect_k
            pred = closest_aprox_eucl_distance(proj, avg_mat)
            print(pred, i)
            if pred == i:
                success += 1
            else:
                fail += 1
    print("success rate: %2f" % (success / (success + fail)))
    print("fail rate: %2f" % (fail / (success + fail)))
    
    
if __name__ == "__main__":
    
    #Samples by subject
    samples = 3
    
    #Number of subjects
    subjects = 40
    
    #Width of .pmg files
    width = 92
    
    #Height of .pmg files
    height = 112
    
    exp = 2
    s = []
    for x in range(1, subjects + 1):
        for j in range(1, samples + 1):
            s.append(load_8_bit_pgm("orl_faces/s" + str(x) + "/"+str(j)+".pgm"))
        
    # I want rows to be subjects
    mat = np.matrix(s)
    
    # Mean of each pixel
    x = [np.mean(mat[:, i]) for i in range(mat.shape[1])]
    
    # Center the matrix
    mat -= x
    
    # Compute kernel matrix K (using k(x, y) = (x * y) ** subjects)
    k = np.matrix(np.empty((subjects * samples, subjects * samples)))
    for i in range(k.shape[0]):
        for j in range(k.shape[1]):
            # polynomial kernel
            k[i, j] = np.power(mat[i, :] * mat[j, :].T, exp)
            k[j, i] = k[i, j]
    
    # Get eigenvectors and eigenvalues of K
    eigval_k, eigvect_k = np.linalg.eig(k)
    eigvect_k /= (eigval_k ** 1/2)
    # Training predictions
    train_pred = k * eigvect_k
    get_kernel_accuracy(eigvect_k, train_pred, x, mat, "orl_faces", subjects, 10, samples, exp)

