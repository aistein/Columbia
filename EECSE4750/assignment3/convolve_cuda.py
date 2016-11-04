#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Matrix Convolution in PyCUDA
"""

import numpy as np
#import PYCUDA modules and libraries
from pycuda import driver, compiler, gpuarray, tools
import sys
#the following module is used to mark the time stamps
import time
#import necessary scipy libraries
import scipy as sp
import scipy.signal
from scipy.signal import convolve2d as conv2d

# -- initialize the device
import pycuda.autoinit

############################
##CUDA KERNEL
###########################

##################################################
##Python Code starts here
##################################################

M = 3 #rows
N = 3 #columns
F = 3 #square dim of "kernel/filter"

# a = np.random.randint(0, 3, (M,N)).astype(np.uint32) #a is matrix which will be convolved
a = np.matrix(' 1 2 3; 4 5 6; 7 8 9')
a_full = np.matrix('0 0 0 0 0 ; 0 1 2 3 0; 0 4 5 6 0 ; 0 7 8 9 0; 0 0 0 0 0')

# f = np.random.randint(0, 2, (F,F)).astype(np.uint32) #f is kernel matrix
f = np.matrix('-1 -2 -1 ; 0 0 0 ; 1 2 1')

# mode='same' gives equal (unpadded) input size to OUTPUT size
# boundary='fill' gives zeros around the input as padding
c = conv2d(a, f, mode='same', boundary='fill')

print "whatever scipy spits out for convolution"
print "a: \n", a
print "f: \n", f
print "c: \n", c

c_test = np.empty_like(a)
print a_full

# implemented my own pythonic convolution to prepare for GPU Kernel deployment
for i in xrange(M+2): # rows
    for j in xrange(N+2): # columns
        # the above will iterate the filter over the whole of a_full (padded)
        if((i > 0 and i < M+2 -1) and (j > 0 and j < N+2 -1)):
        # check to make sure we only operate within the bounds of the OUTPUT
            # adjust the indices we use on OUTPUT w.r.t. padded input
            c_i = i - 1;
            c_j = j - 1;
            for m in xrange(F):
                for n in xrange(F):
                    # need "i-m+1" etc. becuase when i,j = 1,1 ... i-m,j-n = -1 -1
                    c_test[c_i, c_j] += f[m,n] * a_full[i-m+1, j-n+1]
print c_test
