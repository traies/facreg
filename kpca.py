#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  6 14:42:01 2017

@author: traies
"""

import matplotlib.pyplot as plt

import eigen as eigen
import time
from utils.pgm_utils import *
from utils.svm_utils import *
from utils.plot_utils import *

path_faces = "our_faces/"
path_plots = "plots/"

if __name__ == "__main__":
    
    #Base samples
    bsamples = 6
    
    #Samples by subject
    samples = 10
    
    #Number of subjects
    subjects = 5
    
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
            s.append(load_8_bit_pgm(path_faces +"/s" + str(x) + "/"+str(j)+".pgm"))
        
    # I want rows to be subjects
    mat = np.matrix(s)
    
    # Center the matrix
    mean = mat.mean(axis=0)
    mat -= mean

    sta = time.perf_counter()

    # Compute kernel matrix K (using k(x, y) = (x * y) ** exp)
    k = np.power(mat  * mat.T  / trainno + 1 , exp) 
    
    # n aux matrix
    n = 1 / trainno * np.matrix(np.ones([trainno, trainno]))

    # Center k matrix
    k = k - n * k - k * n + n * k * n 

    eigval_k, eigvect_k = eigen.francis(k)

    for i in range(len(eigval_k)):
        eigvect_k[:, i] = eigvect_k[:, i] / np.sqrt(abs(eigval_k[i]))

    end = time.perf_counter()
    print("----")
    print("KPCA")
    print("----")
    print("Tiempo de corrida: {}".format(end - sta))

    # Projection and Clasification
    tests = []
    for x in range(1, subjects + 1):
        for j in range(bsamples+1, samples + 1):
            tests.append(load_8_bit_pgm(path_faces +"/s" + str(x) + "/"+str(j)+".pgm"))
    testm = np.matrix(tests)
    testm -= mean
    
    testp_k = np.power(testm * mat.T  / trainno + 1, exp)
    
    n2 = 1 / trainno * np.matrix(np.ones([testno, trainno]))
    
    testp_k = testp_k - n2 * k - testp_k * n + n2 * k * n 

    trainproj = k * eigvect_k
    testproj = testp_k * eigvect_k

    class_list = [i for i in range(subjects) for j in range(bsamples)]
    testl = [i for i in range(subjects) for j in range(bsamples, samples)]
    g = print_predict_all(trainproj, testproj, class_list, testl)
    print_predict(trainproj, testproj, class_list, testl)

    plot_predict_all(path_plots, g, bsamples, subjects, 'kpca_train')

    