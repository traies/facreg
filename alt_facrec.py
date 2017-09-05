#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  5 10:13:05 2017

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

def get_accuracy_benchmark(eigfaces, train_mat, mean, base_path, subjects, samples, base_samples):
    success = 0
    for i in range(subjects):
        for j in range(samples):
            test = np.matrix(load_8_bit_pgm(base_path + "/s"+str(i+1)+"/"+str(j+1)+".pgm"))
            test -= mean
            proj = test * eigfaces
            pred = closest_aprox_eucl_distance(proj, train_mat)
            if pred // base_samples == i:
                success += 1
    print("success rate: "+ str(success / (subjects * samples)))
    

if __name__ == "__main__":
    
    #Samples by subject
    samples = 3
    
    #Number of subjects
    subjects = 40
    
    #Width of .pmg files
    width = 92
    
    #Height of .pmg files
    height = 112
    
    s = []
    for x in range(1, subjects + 1):
        for j in range(1, samples + 1):
            s.append(load_8_bit_pgm("orl_faces/s" + str(x) + "/"+str(j)+".pgm"))
        
    # I want columns to be subjects
    mat = np.matrix(s)
    
    # Mean of each pixel
    x = [np.mean(mat[:, i]) for i in range(mat.shape[1])]
    
    # Center the matrix
    mat -= x
    
    # svd of the matrix
    u, sigma, vt = np.linalg.svd(mat, full_matrices=False)
    
    # untranspose vector v
    v = vt.T
    
    # principal components
    t = mat * v
    
    # eigenfaces 
    eigenfaces = v * np.diag(sigma)
    
    print(eigenfaces.shape)
    
    # eigenfaces normalization for image
    e = []
    for i in range(eigenfaces.shape[1]):
        a = np.squeeze(np.asarray(eigenfaces[:,i]))
        e.append((a - min(a)) * 255 / (max(a) - min(a)))
    
    # print eigenfaces
    for i in range(subjects * samples):
        save_8_bit_pgm("alt_eigenfaces/eigenface"+str(i)+".pgm", e[i].astype(int), 92, 112)
    
    # Print prediction success rate
    get_accuracy_benchmark(v, t, x, "orl_faces", subjects, 10, samples)

    
    
    
    
    