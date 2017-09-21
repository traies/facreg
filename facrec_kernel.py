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
from utils.directory_utils import *

path_faces = "our_faces/"
path_plots = "plots/"

from sklearn import svm


def predict_all(trainproj, testp,  subjects, samples, base_samples):
    
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
    
    # Compute kernel matrix K (using k(x, y) = (x * y) ** exp)
    k = np.power(mat  * mat.T  / trainno + 1 , exp) 
    
    # n aux matrix
    n = 1 / trainno * np.matrix(np.ones([trainno, trainno]))

    # Center k matrix
    k = k - n * k - k * n + n * k * n 
    
    # Get eigenvectors and eigenvalues of K
    eigval_k1, eigvect_k1 = np.linalg.eigh(k)
    
    sta = time.perf_counter()
    eigval_k, eigvect_k = eigen.francis(k)
    end = time.perf_counter()
    print("tiempo de corrida: {}".format(end - sta))
    print(eigvect_k.shape)
    eigval_k1 = np.flipud(eigval_k1)
    eigvect_k1 = np.fliplr(eigvect_k1)
    
    for i in range(len(eigval_k)):
        print(eigval_k1[i], eigval_k[i], abs(eigval_k[i] - eigval_k1[i]), i)
    
    
    for i in range(len(eigval_k)):
        eigvect_k[:, i] = eigvect_k[:, i] / np.sqrt(abs(eigval_k[i]))
    
    tests = []
    for x in range(1, subjects + 1):
        for j in range(bsamples+1, samples + 1):
            tests.append(load_8_bit_pgm(path_faces +"/s" + str(x) + "/"+str(j)+".pgm"))
    testm = np.matrix(tests)
    testm -= mean
    
    testp_k = np.power(testm * mat.T  / trainno + 1, exp)
    
    n2 = 1 / trainno * np.matrix(np.ones([testno, trainno]))
    
    testp_k = testp_k - n2 * k - testp_k * n + n2 * k * n 
    
    print("a", testp_k.shape)
    trainproj = k * eigvect_k
    testp = testp_k * eigvect_k
    
    g = [[],[]]
    
    for i in range(1, eigvect_k.shape[1]+1):
        aux = predict_all(trainproj[:, 0:i], testp[:, 0:i], subjects, samples, bsamples)
        g[0].append(i)
        g[1].append(aux*100)
        print("using {0} eigenvectors: {1}".format(i, aux))
        
    plt.plot(g[0],g[1])
    plt.suptitle('Porcentaje de acierto según cantidad de autocaras', fontweight='bold')
    plt.ylabel('Acierto (%)')
    plt.xlabel('Autocaras')
    validateDirectory(path_plots)
    plt.savefig(path_plots + 'kpca_train' + str(bsamples) + '_subjects' + str(subjects) + '.png')

    