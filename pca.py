#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  5 10:13:05 2017

@author: traies
"""
import eigen as eigen
import time
from utils.pgm_utils import *
from utils.svm_utils import *
from utils.plot_utils import *

path_faces = "our_faces/"
path_plots = "plots/"
path_mean = "mean/"
path_eigenfaces = "pca_eigenfaces/"


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
    
    trainno = bsamples * subjects
    testno = subjects * (samples - bsamples)
    s = []
    for x in range(1, subjects + 1):
        for j in range(1, bsamples + 1):
            s.append(load_8_bit_pgm(path_faces +"/s" + str(x) + "/"+str(j)+".pgm"))
        
    # I want rows to be subjects
    mat = np.array(s)
    
    # Mean of each pixel
    mean = mat.mean(axis=0)
    
    print(mat)
    # Center the matrix
    mat -= mean
    start = time.perf_counter()
    # Covariance of the matrix
    cov = mat @ mat.T
    
    sta = np.max(cov)
    cov /= sta
    
    print(cov)
    print(cov.shape)
    startF = time.perf_counter()
    eigval, eigvect = eigen.francis(cov)
    endF = time.perf_counter()

    eigvect = 1 / sta *  mat.T @ eigvect 
    eigenfaces =  eigvect @ np.diag(eigval)
    end = time.perf_counter()

    print("-------")
    print("Francis")
    print("-------")
    print("Tiempo de corrida: %.3f segundos" % (endF - startF))

    print("----")
    print("PCA")
    print("----")
    print("Tiempo de corrida: {}".format(end - start))
#    # Eigenfaces normalization for image
    e = []
    for i in range(eigenfaces.shape[1]):
        
        a = np.squeeze(np.asarray(eigenfaces[:,i]))
        e.append((a - min(a)) * maxgrey / (max(a) - min(a)))
    
    # Print eigenfaces
    for i in range(trainno):
        save_8_bit_pgm(path_eigenfaces + "eigenface"+str(i)+".pgm", e[i].astype(int), width, height)

    # Print mean
    printmean = (mean - min(mean)) * maxgrey / (max(mean) - min(mean))
    save_8_bit_pgm(path_mean + "mean.pgm", printmean.astype(int), width, height)

    # Projection and Clasification
    tests = []
    for x in range(1, subjects + 1):
        for j in range(bsamples+1, samples + 1):
            tests.append(load_8_bit_pgm(path_faces +"/s" + str(x) + "/"+str(j)+".pgm"))
    testm = np.array(tests)
    testm -= mean
    
    trainproj = mat @ eigvect
    testproj = testm @ eigvect
    # Print prediction success rate
    class_list = [i for i in range(subjects) for j in range(bsamples)]
    testl = [i for i in range(subjects) for j in range(bsamples, samples)]
    g = print_predict_all(trainproj, testproj, class_list, testl)
    print_predict(trainproj, testproj, class_list, testl)

    plot_predict_all(path_plots, g, bsamples, subjects, 'pca_train')