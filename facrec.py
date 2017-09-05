#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep  2 17:36:50 2017

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

def closest_aprox_man_distance(proj_vect, train_mat):
    a = -1
    mi = np.Infinity
    for i in range(train_mat.shape[1]):
        m = np.sum(abs(train_mat[:,i] - proj_vect))
        if m <= mi:
            mi = m
            a = i
    return a

def closest_aprox_eucl_distance(proj_vect, train_mat):
    a = -1
    mi = np.Infinity
    for i in range(train_mat.shape[1]):
        m = np.sum(np.power(train_mat[:,i] - proj_vect, 2))
        if m <= mi:
            mi = m
            a = i
    return a

def get_accuracy_benchmark(eigfaces, train_mat, mean, base_path, subjects, samples):
    success = 0
    for i in range(subjects):
        for j in range(samples):
            test = np.matrix(load_8_bit_pgm(base_path + "/s"+str(i+1)+"/"+str(j+1)+".pgm"))
            test -= mean
            proj = eigfaces.T * test.T
            pred = closest_aprox_eucl_distance(proj, train_mat)
            if pred == i:
                success += 1
    print("success rate: "+ str(success / (subjects * samples)))
    return 

if __name__ == "__main__":
    
    #Number of subjects
    subjects = 40
    
    #Width of .pmg files
    width = 92
    
    #Height of .pmg files
    height = 112
    
    s = []
    for x in range(1, subjects + 1):
        s.append(load_8_bit_pgm("orl_faces/s" + str(x) + "/1.pgm"))
    # I want columns to be subjects
    mat = np.matrix(s).transpose()
    # Mean of each pixel
    x = [np.mean(mat[i, :]) for i in range(mat.shape[0])]
    
    x = np.repeat(x, subjects).reshape((width * height, subjects))
    
    og = np.copy(mat)
    # Center the matrix
    mat -= x
    
    # svd of the matrix
    u, sigma, vt = np.linalg.svd(mat, full_matrices=False)
    s = u * np.diag(sigma)
    
    # u = eigenvectors of the centered matrix
    eigenfaces = []    
    for i in range(u.shape[1]):
        t = np.squeeze(np.asarray(u[:,i]))
        eigenfaces.append((t - min(t)) * 255 / (max(t) - min(t)))
    
    e = np.matrix(eigenfaces).T
    e = e.astype(int)
    xi = x.astype(int)
    save_8_bit_pgm("mean.pgm", [int(i) for i in np.squeeze(np.asarray(xi[:,0]))], 92, 112)
    
    for i in range(40):
        save_8_bit_pgm("eigenfaces/eigenface"+str(i)+".pgm", np.squeeze(np.asarray(e[:, i])), 92,112)
    
    # Initial projections on face space
    t = u.T * mat 
    
    # Print prediction success rate
    get_accuracy_benchmark(u, t, x[:, 0], "orl_faces", 40, 10)
    
    # compression test
#    acc = 10
#    c = u[:, :acc] * np.diag(sigma[:acc]) * vt[:acc, :] + x
#    c = np.clip(c, 0, 255)
#    c = c.astype(int)
#    
#    for i in range(40):
#        save_8_bit_pgm("compressed/comp"+str(i)+".pgm", np.squeeze(np.asarray(c[:, i])), 92,112)
#    

