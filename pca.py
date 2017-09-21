#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  5 10:13:05 2017

@author: traies
"""
from sklearn import svm
import eigen as eigen
import time
import matplotlib.pyplot as plt
from utils.pgm_utils import *
from utils.directory_utils import *

path_faces = "our_faces/"
path_plots = "plots/"
path_mean = "mean/"
path_eigenfaces = "pca_eigenfaces/"


def predict_all(trainproj, testproj, class_list, testl):
    
    clf = svm.LinearSVC( random_state=0)
    clf.fit(trainproj, class_list)
    
    
    return clf.score(testproj, testl)
    
    
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
    
    # Center the matrix
    mat -= mean

    # Covariance of the matrix
    cov = 1 / (trainno - 1) *  mat @ mat.T
    sta = time.perf_counter()
    eigval, eigvect = eigen.francis(cov)
    end = time.perf_counter()
    print("tiempo de corrida: {}".format(end - sta))
    
    eigvect = 1 / (trainno - 1) * mat.T @ eigvect 
    eigenfaces =  eigvect @ np.diag(eigval)
    
    # Eigenfaces normalization for image
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
    g = [[],[]]
    for i in range(1, trainproj.shape[1] + 1):
        aux = predict_all(trainproj[:, 0:i], testproj[:, 0:i], class_list, testl)
        g[0].append(i)
        g[1].append(aux*100)
        print("using {0} eigenvectors: {1}".format(i, aux))
        
    plt.plot(g[0],g[1])
    plt.suptitle('Porcentaje de acierto según cantidad de autocaras',fontweight='bold')
    plt.ylabel('Acierto (%)')
    plt.xlabel('Autocaras')
    validateDirectory(path_plots)
    plt.savefig(path_plots + 'pca_train' + str(bsamples) + '_subjects' + str(subjects) + '.png')
    