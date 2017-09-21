#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  5 10:13:05 2017

@author: traies
"""
import numpy as np
from sklearn import svm
import eigen as my_svd
import time
import matplotlib.pyplot as plt

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

def get_accuracy_benchmark(eigfaces, train_mat, mean, base_path, subjects, samples, base_samples):
    # get averages matrix
    avg_mat = []
    for i in range(subjects):
        subj_list = []
        for j in range(base_samples):
            subj = np.matrix(load_8_bit_pgm(base_path + "/s"+str(i+1)+"/"+str(j+1)+".pgm"))
            subj -= mean
            proj = subj * eigfaces
            subj_list.append(np.squeeze(np.asarray(proj)))
        subj_list_m = np.matrix(subj_list)
        x = [np.mean(subj_list_m[:, i]) for i in range(subj_list_m.shape[1])]
        avg_mat.append(x)
    avg_mat = np.matrix(avg_mat)
    
    
    success = 0
    fail = 0
    for i in range(subjects):
        for j in range(base_samples, samples):
            test = np.matrix(load_8_bit_pgm(base_path + "/s"+str(i+1)+"/"+str(j+1)+".pgm"))
            test -= mean
            proj = test * eigfaces
            pred = closest_aprox_eucl_distance(proj, avg_mat)
            if pred == i:
                success += 1
            else:
                fail += 1
    print("success rate: %2f" % (success / (success + fail)))
    print("fail rate: %2f" % (fail / (success + fail)))
    

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
            s.append(load_8_bit_pgm("our_faces/s" + str(x) + "/"+str(j)+".pgm"))
        
    # I want rows to be subjects
    mat = np.array(s)
    
    # Mean of each pixel
    mean = mat.mean(axis=0)
    
    # Center the matrix
    mat -= mean
#    u, sigma, vt = np.linalg.svd(mat, full_matrices=False)
    # cov of the matrix
    cov = 1 / (trainno - 1) *  mat @ mat.T
    sta = time.perf_counter()
    eigval, eigvect = my_svd.francis(cov)
    end = time.perf_counter()
    print("tiempo de corrida: {}".format(end - sta))
    
    eigvect = 1 / (trainno - 1) * mat.T @ eigvect 
    eigenfaces =  eigvect @ np.diag(eigval)
    
    # eigenfaces normalization for image
    e = []
    for i in range(eigenfaces.shape[1]):
        
        a = np.squeeze(np.asarray(eigenfaces[:,i]))
        e.append((a - min(a)) * 255 / (max(a) - min(a)))
    
    # print eigenfaces
    for i in range(trainno):
        save_8_bit_pgm("pca_eigenfaces/eigenface"+str(i)+".pgm", e[i].astype(int), 92, 112)
    
    printmean = (mean - min(mean)) * 255 / (max(mean) - min(mean))
    save_8_bit_pgm("mean/mean.pgm", printmean.astype(int), 92, 112)
    tests = []
    for x in range(1, subjects + 1):
        for j in range(bsamples+1, samples + 1):
            tests.append(load_8_bit_pgm("our_faces/s" + str(x) + "/"+str(j)+".pgm"))
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
    #plt.show()
    plt.suptitle('Porcentaje de acierto según cantidad de autocaras',fontweight='bold')
    plt.ylabel('Acierto (%)')
    plt.xlabel('Autocaras')
    plt.savefig('plots/orl_b' + str(bsamples) + '_s' + str(subjects) + '.png')
    