#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  6 14:42:01 2017

@author: traies
"""

import matplotlib.pyplot as plt

import numpy as np
import svd as svd
import time

from sklearn import svm

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
            arr[i] = int(ord(img.read(1))) / 255
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

def get_kernel_accuracy(eigvect_k, mean, og_mat, base_path, subjects, samples, base_samples, exp):
    
    # get averages matrix
    avg_mat = []
    for i in range(subjects):
        subj_list = []
        for j in range(base_samples):
            subj = np.matrix(load_8_bit_pgm(base_path + "/s"+str(i+1)+"/"+str(j+1)+".pgm"))
            subj -= mean
            proj = np.power(subj * og_mat.T, exp) * eigvect_k.T
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
            proj = np.power(test * og_mat.T, exp) * eigvect_k.T
            pred = closest_aprox_eucl_distance(proj, avg_mat)
            if pred == i:
                success += 1
            else:
                fail += 1
    print("success rate: %2f" % (success / (success + fail)))
    print("fail rate: %2f" % (fail / (success + fail)))

def predict_all(trainproj, testp,  base_path, subjects, samples, base_samples):
    
    class_list = [i for i in range(subjects) for j in range(base_samples)]
    clf = svm.LinearSVC(random_state=0)
    clf.fit(trainproj, class_list)
    
    testl = [i for i in range(subjects) for j in range(base_samples, samples)]
    return clf.score(testp, testl)


if __name__ == "__main__":
    
    #Base samples
    bsamples = 6
    
    #Samples by subject
    samples = 10
    
    #Number of subjects
    subjects = 40
    
    #Width of .pmg files
    width = 92
    
    #Height of .pmg files
    height = 112
    
    exp = 2
    s = []
    
    trainno = bsamples * subjects
    testno = subjects * (samples - bsamples)
    for x in range(1, subjects + 1):
        for j in range(1, bsamples + 1):
            s.append(load_8_bit_pgm("orl_faces/s" + str(x) + "/"+str(j)+".pgm"))
        
    # I want rows to be subjects
    mat = np.matrix(s)
    
    # Center the matrix
    mean = mat.mean(axis=0)
    mat -= mean
    
    # Compute kernel matrix K (using k(x, y) = (x * y) ** exp)
    k = np.power(mat  * mat.T  / trainno + 1 , exp) 
    
    # n aux matrix
    n = 1 / trainno * np.matrix(np.ones([trainno, trainno]))

    # Center k matrix
    k = k - n * k - k * n + n * k * n 
    
    # Get eigenvectors and eigenvalues of K
    eigval_k1, eigvect_k1 = np.linalg.eigh(k)
    
    sta = time.perf_counter()
    eigval_k, eigvect_k = svd.francis(k)
    end = time.perf_counter()
    print("tiempo de corrida: {}".format(end - sta))
    
    
    eigval_k = np.flipud(eigval_k)
    eigvect_k = np.fliplr(eigvect_k)
    
    for i in range(len(eigval_k)):
        print(eigval_k1[i], eigval_k[i], abs(eigval_k[i] - eigval_k1[i]), i)
    
    exit(0)
    for i in range(len(eigval_k)):
        eigvect_k[:, i] = eigvect_k[:, i] / np.sqrt(abs(eigval_k[i]))
    
    tests = []
    for x in range(1, subjects + 1):
        for j in range(bsamples+1, samples + 1):
            tests.append(load_8_bit_pgm("orl_faces/s" + str(x) + "/"+str(j)+".pgm"))
    testm = np.matrix(tests)
    testm -= mean
    
    testp_k = np.power(testm * mat.T  / trainno + 1, exp)
    
    n2 = 1 / trainno * np.matrix(np.ones([testno, trainno]))
    
    testp_k = testp_k - n2 * k - testp_k * n + n2 * k * n 
    trainproj = k * eigvect_k
    testp = testp_k * eigvect_k
    
    g = [[],[]]
    
    for i in range(1, eigvect_k.shape[1]+1):
        aux = predict_all(trainproj[:, 0:i], testp[:, 0:i], "orl_faces", subjects, samples, bsamples)
        g[0].append(i)
        g[1].append(aux*100)
        print("using {0} eigenvectors: {1}".format(i, aux))
        
    plt.plot(g[0],g[1])
    #plt.show()
    plt.suptitle('Prediction accuracy depending on the number of eigenvectors',fontweight='bold')
    plt.ylabel('accuracy (%)')
    plt.xlabel('eigenvectors')
    plt.savefig('plots/orl_b' + str(bsamples) + '_s' + str(subjects) + '.png')

    