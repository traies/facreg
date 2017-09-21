#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  21 19:15:04 2017

@author: marlanti
"""
import numpy as np
from .directory_utils import *

maxgrey = 255

def save_8_bit_pgm(filepath, arr, width, height):
    validateDirectory(filepath)
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
            arr[i] = int(ord(img.read(1))) / maxgrey
    return arr