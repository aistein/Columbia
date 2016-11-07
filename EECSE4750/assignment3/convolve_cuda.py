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

        # if ( threadIdx.x == 0 && j != 0 ) {
        # // vertical left block edges
        #     DS_A_PAD[threadIdx.y*(N_LIM+2)+threadIdx.x] = A[i*N + (j-1)];
        # }
        # if ( threadIdx.x == N_LIM-1 && j != N-1 ) {
        # // veritcal right block edges
        #     DS_A_PAD[(threadIdx.y+1)*(N_LIM+2)+(threadIdx.x+2)] = A[i*N + (j+1)];
        # }
        # if ( threadIdx.y == 0 && i != 0 ) {
        # // horizontal top block edges
        #     DS_A_PAD[threadIdx.y*(N_LIM+2)+threadIdx.x] = A[(i-1)*N + j];
        # }
        # if ( threadIdx.y == M_LIM-1 && i != M-1 ) {
        # // horizontal bottom block edges
        #     DS_A_PAD[(threadIdx.y+2)*(N_LIM+2)+(threadIdx.x+1)] = A[(i+1)*N + j];
        # }
        # // internal elements

kernel_code_template = """
//2D Convolution function
__global__ void convolve2d(unsigned int* A, unsigned int* K,
                           const unsigned int M, const unsigned int N,
                           const unsigned int F, unsigned int* C)
{

    #ifndef TILE_M
        #define TILE_M 3u
    #endif
    #ifndef TILE_N
        #define TILE_N 3u
    #endif

    // A - input size MxN
    // K - Kernel size FxF
    // M - Row Dim of UN-Padded A
    // N - Column Dim of UN-Padded A
    // F - Square Dim of K (Pitch)
    // C - Output size MxN

    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;

    if(i < M && j <  N){
    // check to make sure we are within bounds of the overall output size

        // create flattened padded matrix size M+2 x N+2 to use for convolution input
        extern __shared__ unsigned int DS_A_PAD[];
        unsigned int M_LIM = (M > 32) ? ((blockIdx.x+1)*blockDim.x < M) ? blockDim.x : blockDim.x - (blockIdx.x+1)*blockDim.x % M : M;
        unsigned int N_LIM = (N > 32) ? ((blockIdx.y+1)*blockDim.y < N) ? blockDim.y : blockDim.y - (blockIdx.y+1)*blockDim.y % N : N;
        for(unsigned int k = 0; k < M_LIM + 2; k++){
            for(unsigned int l = 0; l < N_LIM + 2; l++){
                DS_A_PAD[k*(N_LIM+2) + l] = 0;
            }
        }

        __syncthreads();

        if ( threadIdx.y == 0 && j != 0 ) {
        // vertical left block edges - CHECK!
            DS_A_PAD[(threadIdx.x+1)*(N_LIM+2)+threadIdx.y] = A[i*N + (j-1)];
        }
        if ( threadIdx.y == N_LIM-1 && j != N-1 ) {
        // veritcal right block edges - CHECK!
            DS_A_PAD[(threadIdx.x+1)*(N_LIM+2)+(threadIdx.y+2)] = A[i*N + (j+1)];
        }
        if ( threadIdx.x == 0 && i != 0 ) {
        // horizontal top block edges - CHECK!
            DS_A_PAD[threadIdx.x*(N_LIM+2)+(threadIdx.y+1)] = A[(i-1)*N + j];
        }
        if ( threadIdx.x == M_LIM-1 && i != M-1 ) {
        // horizontal bottom block edges - CHECK!
            DS_A_PAD[(threadIdx.x+2)*(N_LIM+2)+(threadIdx.y+1)] = A[(i+1)*N + j];
        }
        if ( threadIdx.x == M_LIM-1 && threadIdx.y == N_LIM-1 && j != N-1 && i != M-1 ) {
        // lower righthand corners (outside current block) - CHECK!
            DS_A_PAD[(threadIdx.x+2)*(N_LIM+2)+(threadIdx.y+2)] = A[(i+1)*N + (j+1)];
        }
        if ( threadIdx.x == M_LIM-1 && threadIdx.y == 0 && j != 0 ) {
        // lower lefthand corners (outside current block) - CHECK!
            DS_A_PAD[(threadIdx.x+2)*(N_LIM+2)+threadIdx.y] = A[(i+1)*N + (j-1)];
        }
        if ( threadIdx.x == 0 && threadIdx.y == N_LIM-1 && i != 0 ) {
        // upper righthand corners (outside current block) - CHECK!
            DS_A_PAD[threadIdx.x*(N_LIM+2)+(threadIdx.y+2)] = A[(i-1)*N + (j+1)];
        }
        if ( threadIdx.x == 0 && threadIdx.y == 0 && i != 0 && j != 0 ) {
        // upper lefthand corners (outside current block)
            //DS_A_PAD[(threadIdx.x+1)*(N_LIM+2)+threadIdx.y] = A[i*N + (j-1)];
            //DS_A_PAD[threadIdx.x*(N_LIM+2)+(threadIdx.y+1)] = A[(i-1)*N + j];
            //DS_A_PAD[threadIdx.x*(N_LIM+2)+threadIdx.y] = A[(i-1)*N + (j-1)];
        }
        // internal elements
          DS_A_PAD[(threadIdx.x+1)*(N_LIM+2)+(threadIdx.y+1)] = A[i*N + j];

        __syncthreads();

        C[i*N + j] = DS_A_PAD[(threadIdx.x+1)*(N_LIM+2) + (threadIdx.y+1)];
        //C[i*N + j] = i*N + j;

        //// Convolution Calculation for element (i,j)
        //C[i*N + j] = 0;
        //for(unsigned int m = 0; m < F; m++){
        //    for(unsigned int n = 0; n < F; n++){
        //        //C[i*N + j] += K[m*F + n] * DS_A_PAD[((i+1)-m+1)*(N_LIM+2) +((j+1)-n+1)];
        //        C[i*N + j] += K[m*F + n] * DS_A_PAD[((threadIdx.x+1)-m+1)*(N_LIM+2) +((threadIdx.y+1)-n+1)];
        //        //C[i*N +j] = 0;
        //    }
        //}// end convolution calculation

        __syncthreads();

    }// end output boundary check
}
"""

##################################################
##Python Code starts here
##################################################

# Configurations
M = 33 #rows
N = 33 #columns
F = 3 #square dim of "kernel/filter"

a = np.random.randint(0, 9, (M,N)).astype(np.uint32) #a is matrix which will be convolved

# create an FxF filter of random numbers
# f = np.random.randint(0, 9, (F,F)).astype(np.uint32) #f is kernel matrix
f = np.ones((F,F)).astype(np.uint32) #f is kernel matrix

# mode='same' gives equal (unpadded) input size to OUTPUT size
# boundary='fill' gives zeros around the input as padding
c = conv2d(a, f, mode='same', boundary='fill')

print "====================== PART 1 =========================="
print "--------------------- PYTHON ---------------------------"
print "a: \n", a
print "f: \n", f
print "c: \n", c

print "----------------------- CUDA ----------------------------"

# get the kernel code from the template
kernel_code = kernel_code_template

# compile the kernel code, with options
TILE_M = str(M) + 'u' if (M <= 32) else str(32) + 'u'
TILE_N = str(N) + 'u' if (N <= 32) else str(32) + 'u'
options = "-DTILE_M=" + TILE_M + " -DTILE_N=" + TILE_N
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
unit_size = np.dtype(np.uint32).itemsize
shared_mem = (M+2)*(N+2)*unit_size if (M <= 32) and (N <= 32) else 34*34*unit_size
conv(A_buf,K_buf,np.uint32(M),np.uint32(N),np.uint32(F),C_buf,block = (32,32,1),grid = (np.uint32(M-1/32)+1,np.uint32(N-1/32)+1,1),shared=shared_mem)
# conv(A_buf,K_buf,np.uint32(M),np.uint32(N),np.uint32(F),C_buf,block = (M,N,1),grid = (1,1,1),shared=(M+2)*(N+2)*np.dtype(np.uint32).itemsize)

# copy data back from GPU
C_gpu = C_buf.get()
print "c_gpu: \n", C_gpu
