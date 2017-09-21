#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  21 19:39:03 2017

@author: marlanti
"""
import os

def validateDirectory(filepath):
    path, file = filepath.split("/")
    if not os.path.exists(path):
        os.makedirs(path)