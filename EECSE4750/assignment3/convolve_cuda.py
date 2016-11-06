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
############################

    # if ( i == 0 && j == 0){
    # // upper left corner
    #     DS_A_PAD[0,0] = 0;
    #     DS_A_PAD[0,1] = 0;
    #     DS_A_PAD[1,0] = 0;
    #     DS_A_PAD[1,1] = A[i*A_N + j];
    # } else if ( i == 0 && j < A_N-1 ) {
    # // top row
    #     DS_A_PAD[0,j] = 0;
    #     DS_A_PAD[1,j] = A[i*A_N + j];
    # } else if ( i == 0 && j == A_N-1 ) {
    # // upper right corner
    #     DS_A_PAD[0,A_N] = 0;
    #     DS_A_PAD[0,j] = 0;
    #     DS_A_PAD[1,A_N] = 0;
    #     DS_A_PAD[1,j] = A[i*A_N + j];
    # } else if ( i < A_M-1 && j == 0 ) {
    # // left coluA_Mn
    #     DS_A_PAD[i,0] = 0;
    #     DS_A_PAD[i,1] = A[i*A_N + j];
    # } else if ( i == A_M-1 && j == 0 ) {
    # // bottoA_M left corner
    #     DS_A_PAD[A_M,0] = 0;
    #     DS_A_PAD[i,0] = 0;
    #     DS_A_PAD[A_M,1] = 0;
    #     DS_A_PAD[i,1] = A[i*A_N + j];
    # } else if ( i == A_M-1 && j < A_N-1 ) {
    # // bottoA_M row
    #     DS_A_PAD[A_M,j] = 0;
    #     DS_A_PAD[i,j] = A[i*A_N +j];
    # } else if ( i == A_M-1 && j == A_N-1 ) {
    # // bottoA_M right corner
    #     DS_A_PAD[A_M,A_N] = 0;
    #     DS_A_PAD[i,A_N] = 0;
    #     DS_A_PAD[A_M,j] = 0;
    #     DS_A_PAD[i,j] = A[i*A_N + j];
    # } else {
    # // not on any edge or corner
    #     DS_A_PAD[i+1,j+1] = A[i*A_N +j];
    # }
    # __syncthreads();

    # unsigned int a_index = i*A_N + j;
    # if ( a_index == 0 ){
    # // pad top row, update first element
    #     for(unsigned int l=0; l<A_N+3; l++){
    #         DS_A_PAD[l] = 0;
    #     }
    #     DS_A_PAD[A_N+3] = A[a_index];
    # } else if ( a_index == M*N-1 ) {
    # // pad bottom row, update last element
    #     for(unsigned int l=0; l<A_N+3; l++){
    #         DS_A_PAD[(A_M+2)*(A_N+1) + l] = 0;
    #     }
    #     DS_A_PAD[(A_M+2)*(A_N+1) - 1] = A[a_index];
    # } else if (j == A_N-1){
    # // all right edge valid and padded elements
    #     DS_A_PAD[(i+2)*(A_N+2)+j+1] = A[a_index];
    #     DS_A_PAD[(i+2)*(A_N+2)+j+2] = 0;
    #     DS_A_PAD[(i+2)*(A_N+2)+j+3] = 0;
    # } else {
    # // left edge or completely interior elements
    #     DS_A_PAD[(i+2)*(A_N+2)+j+1] = A[a_index];
    # }

kernel_code_template = """
//2D Convolution function
__global__ void convolve2d(unsigned int* A, unsigned int* K,
                           const unsigned int M, const unsigned int N,
                           const unsigned int F, unsigned int* C)
{

    #ifndef A_M
        #define A_M 3u
    #endif
    #ifndef A_N
        #define A_N 3u
    #endif

    // A - input size MxN
    // K - Kernel size FxF
    // M - Row Dim of UN-Padded A
    // N - Column Dim of UN-Padded A
    // F - Square Dim of K (Pitch)
    // C - Output size MxN

    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;

    // create flattened padded matrix size M+2 x N+2 to use for convolution input
    extern __shared__ unsigned int DS_A_PAD[];
    for(unsigned int k = 0; k < A_M + 2; k++){
        for(unsigned int l = 0; l < A_N + 2; l++){
            DS_A_PAD[k*(A_N+2) + l] = 0;
        }
    }
    __syncthreads();

    DS_A_PAD[(i+1)*(A_N+2)+(j+1)] = A[i*A_N + j];

    __syncthreads();

    C[i*N + j] = DS_A_PAD[(i+1)*(A_N+2) + (j+1)];

    // check to make sure we only operate within the bounds of the OUTPUT
    //if((i>0 && i< M+2-1) && (j>0 && j< N+2-1)){
    //    // adjust the indices we use on OUTPUT w.r.t. padded input
    //    unsigned int c_i = i - 1;
    //    unsigned int c_j = j - 1;
    //    for(unsigned int m = 0; m < F; m++){
    //        for(unsigned int n = 0; n < F; n++){
    //            // need "i-m+1" etc. because when i,j = 1,1 ... i-m,j-n = -1,-1
    //            C[c_i*N + c_j] += K[m*F +n] * A[(i-m+1)*(N+2) + (j-n+1)];
    //        }// end n filter dim
    //    }// end m filter dim
    //}// end boundary check
    //__syncthreads();
}
"""

##################################################
##Python Code starts here
##################################################

# Configurations
M = 4 #rows
N = 3 #columns
F = 3 #square dim of "kernel/filter"
DEBUG = True # print debug statements

a = np.random.randint(0, 9, (M,N)).astype(np.uint32) #a is matrix which will be convolved

# create a padded matrix to pass into the kernel
a_pad = np.empty([M+2, N+2]).astype(np.uint32)
for i in xrange(M+2): # rows
    for j in xrange(N+2): #columns
        if (i > 0 and i < M+2 - 1) and (j > 0 and j < N+2 - 1):
            a_pad[i,j] = a[i-1,j-1]
        else:
            a_pad[i,j] = 0

# create an FxF filter of random numbers
f = np.random.randint(0, 9, (F,F)).astype(np.uint32) #f is kernel matrix

# mode='same' gives equal (unpadded) input size to OUTPUT size
# boundary='fill' gives zeros around the input as padding
c = conv2d(a, f, mode='same', boundary='fill')

print "====================== PART 1 =========================="
print "--------------------- PYTHON ---------------------------"
print ("a: ", a) if DEBUG else ''
print ("a_f: ", a_pad) if DEBUG else ''
print ("f: ", f) if DEBUG else ''
print ("c: ", c) if DEBUG else ''

# implemented my own pythonic convolution to prepare for GPU Kernel deployment
c_test = np.empty_like(a)
for i in xrange(M+2): # rows
    for j in xrange(N+2): # columns
        # the above will iterate the filter over the whole of a_pad (padded)
        if((i > 0 and i < M+2 -1) and (j > 0 and j < N+2 -1)):
        # check to make sure we only operate within the bounds of the OUTPUT
            # adjust the indices we use on OUTPUT w.r.t. padded input
            c_i = i - 1;
            c_j = j - 1;
            for m in xrange(F):
                for n in xrange(F):
                    # need "i-m+1" etc. becuase when i,j = 1,1 ... i-m,j-n = -1 -1
                    c_test[c_i, c_j] += f[m,n] * a_pad[i-m+1, j-n+1]
print ("c manual: ", c_test) if DEBUG else ''

print "----------------------- CUDA ----------------------------"

# get the kernel code from the template
kernel_code = kernel_code_template

# compile the kernel code, with options
A_M = str(M) + 'u'
A_N = str(N) + 'u'
options = "-DA_M=" + A_M + " -DA_N=" + A_N
OPTIONS = [_flag.strip() for _flag in options.split() if _flag.strip()]
print OPTIONS
mod = compiler.SourceModule(kernel_code,options=OPTIONS)

# get the kernel function from the compiled module
conv = mod.get_function("convolve2d")

# create buffers for transfer into GPU
A_buf = gpuarray.to_gpu(a)
K_buf = gpuarray.to_gpu(f)
C_buf = gpuarray.empty((M,N),a.dtype)
C_gpu = np.empty((M,N),a.dtype)

# call to conv
# conv(A_buf,K_buf,np.uint32(M),np.uint32(N),np.uint32(F),C_buf,block = (32,32,1),grid = (np.uint32(M/32)+1,np.uint32(N/32)+1,1))
# conv(A_buf,K_buf,np.uint32(M),np.uint32(N),np.uint32(F),C_buf,block = (3,3,1),grid = (np.uint32(M/3)+1,np.uint32(N/3)+1,1))
conv(A_buf,K_buf,np.uint32(M),np.uint32(N),np.uint32(F),C_buf,block = (M,N,1),grid = (1,1,1),shared=(M+2)*(N+2)*np.dtype(np.uint32).itemsize)

# copy data back from GPU
C_gpu = C_buf.get()
print ("c_gpu: ", C_gpu) if DEBUG else ''
