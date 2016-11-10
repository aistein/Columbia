#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Matrix Convolution in PyCUDA
"""

import numpy as np
np.set_printoptions(threshold=np.nan)
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

kernel_code_template = """
//2D Convolution function
__global__ void convolve2d(int* A, int* K,
                           const int M, const int N,
                           const int F, int* C)
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

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int i = blockIdx.x * blockDim.x + tx;
    int j = blockIdx.y * blockDim.y + ty;

    if(i < M && j <  N){
    // check to make sure we are within bounds of the overall output size

        // create flattened padded matrix size M+2 x N+2 to use for convolution input
        extern __shared__ int DS_A_PAD[];
        int M_LIM = (M > 32) ? ((blockIdx.x+1)*blockDim.x < M) ? blockDim.x : blockDim.x - (blockIdx.x+1)*blockDim.x % M : M;
        int N_LIM = (N > 32) ? ((blockIdx.y+1)*blockDim.y < N) ? blockDim.y : blockDim.y - (blockIdx.y+1)*blockDim.y % N : N;

        //for(int k = 0; k < M_LIM + 2; k++){
            //for(int l = 0; l < N_LIM + 2; l++){
                //DS_A_PAD[k*(N_LIM+2) + l] = 0;
            //}
        //}

        //__syncthreads();

        //// ZERO PADDING - comes before fill so as not to overwrite

        // I (row) interior matrix boundary
        if ( i == M-1 ){
            DS_A_PAD[(tx+2)*(N_LIM+2) + ty] = 0;
            DS_A_PAD[(tx+2)*(N_LIM+2) + (ty+1)] = 0;
            DS_A_PAD[(tx+2)*(N_LIM+2) + (ty+2)] = 0;
        } if ( i == 0 ){
            DS_A_PAD[tx*(N_LIM+2) + ty] = 0;
            DS_A_PAD[tx*(N_LIM+2) + (ty+1)] = 0;
            DS_A_PAD[tx*(N_LIM+2) + (ty+2)] = 0;
        }

        // Matrix Corner boundaries
        if (i == 0 && j == 0){
            DS_A_PAD[0] = 0;
        }
        else if (i == M-1 && j == 0){
            DS_A_PAD[(tx+1)*N_LIM] = 0;
        }
        else if (i == 0 && j == N-1){
            DS_A_PAD[ty+1] = 0;
        }
        else if (i == M-1 && j == N-1){
            DS_A_PAD[(tx+1)*N_LIM + (ty+1)] = 0;
        }

        __syncthreads();

        //// FILL VALUES OF DS_A_PAD

        bool left = ( ty == 0 ) ? true : false;
        bool right = ( ty == N_LIM-1 ) ? true : false;
        bool top = ( tx == 0 ) ? true : false;
        bool bottom = ( tx == M_LIM-1 ) ? true : false;

        if ( left && j != 0 ) {
        // vertical left block edges - CHECK!
            DS_A_PAD[(tx+1)*(N_LIM+2)+ty] = A[i*N + (j-1)];
        }

        if ( right && j != N-1 ) {
        // veritcal right block edges - CHECK!
            DS_A_PAD[(tx+1)*(N_LIM+2)+(ty+2)] = A[i*N + (j+1)];
        }

        if ( top && i != 0 ) {
        // horizontal top block edges - CHECK!
            DS_A_PAD[tx*(N_LIM+2)+(ty+1)] = A[(i-1)*N + j];
        }

        if ( bottom && i != M-1) {
        // horizontal bottom block edges - CHECK!
            DS_A_PAD[(tx+2)*(N_LIM+2)+(ty+1)] = A[(i+1)*N + j];
        }

        if ( bottom && right && j != N-1 && i != M-1 ) {
        // lower righthand block corners (outside current block) - CHECK!
            DS_A_PAD[(tx+2)*(N_LIM+2)+(ty+2)] = A[(i+1)*N + (j+1)];
        }

        if ( bottom && left && j != 0 && i < M-1 ) {
        // lower lefthand block corners (outside current block) - CHECK!
            DS_A_PAD[(tx+2)*(N_LIM+2)+ty] = A[(i+1)*N + (j-1)];
        }

        if ( top && right && i != 0 && j < N-1 ) {
        // upper righthand block corners (outside current block) - CHECK!
            DS_A_PAD[tx*(N_LIM+2)+(ty+2)] = A[(i-1)*N + (j+1)];
        }

        if ( top && left && i != 0 && j != 0 ) {
        // upper lefthand block corners (outside current block)
            DS_A_PAD[tx*(N_LIM+2)+ty] = A[(i-1)*N + (j-1)];
        }

        // J (col) interior matrix boundary
        if ( j == N-1 ){
            DS_A_PAD[tx*(N_LIM+2) + (ty+2)] = 0;
            DS_A_PAD[(tx+1)*(N_LIM+2) + (ty+2)] = 0;
            DS_A_PAD[(tx+2)*(N_LIM+2) + (ty+2)] = 0;
        } if ( j == 0 ){
            DS_A_PAD[tx*(N_LIM+2) + ty] = 0;
            DS_A_PAD[(tx+1)*(N_LIM+2) + ty] = 0;
            DS_A_PAD[(tx+2)*(N_LIM+2) + ty] = 0;
        }

        // internal elements
            DS_A_PAD[(tx+1)*(N_LIM+2)+(ty+1)] = A[i*N + j];

        __syncthreads();

        //// CONVOLUTION Calculation for element (i,j)

        C[i*N + j] = 0;
        for(int m = 0; m < F; m++){
            for(int n = 0; n < F; n++){
                C[i*N + j] += K[m*F + n] * DS_A_PAD[((tx+1)-m+1)*(N_LIM+2) +((ty+1)-n+1)];
            }
        }// end convolution calculation

        __syncthreads();

    }// end output boundary check
}
"""

##################################################
##Python Code starts here
##################################################

# Configurations
M = 69 #rows
N = 69 #columns
F = 3 #square dim of "kernel/filter"

a = np.random.randint(0,10, (M,N)).astype(np.int32) #a is matrix which will be convolved

# create an FxF filter of random numbers
# f = np.random.randint(-1, 5, (F,F)).astype(np.int32) #f is kernel matrix
# f = np.ones((F,F)).astype(np.int32) #f is kernel matrix
f = np.array([[-1., -2. , -1.], [0., 0., 0.], [1., 2., 1.]]).astype(np.int32)

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

# D_buf = gpuarray.empty((M+2,N+2), a.dtype)
# D_gpu = np.empty((M+2,N+2),a.dtype)

# call to conv
unit_size = np.dtype(np.int32).itemsize
shared_mem = (M+2)*(N+2)*unit_size if (M <= 32) and (N <= 32) else 34*34*unit_size
conv(A_buf,K_buf,np.int32(M),np.int32(N),np.int32(F),C_buf,block = (32,32,1),grid = (np.int32(M-1/32)+1,np.int32(N-1/32)+1,1),shared=shared_mem)

# copy data back from GPU
C_gpu = C_buf.get()
# D_gpu = D_buf.get()
print "c_gpu: \n", C_gpu
print "c_gpu == c_cpu ? --> ", np.array_equal(C_gpu,c)
for i in range(M):
    for j in range(N):
        if C_gpu[i][j] != c[i][j] :
            print "element[" + str(i) + "][" + str(j) + "] isn't matching"
            print "C_gpu["+str(i)+"]["+str(j)+"] = ", C_gpu[i][j]
            print "c[i][j] =", c[i][j]
            for p in range(3):
                for q in range(3):
                    print a[i-1+p][j-1+q]

# print "d_gpu: \n", D_gpu
# for i in range(M+2):
#     for j in range(N+2):
#         if (i < M and j < N) and (i > 0 and j > 0):
#             if D_gpu[i][j] != a[i-1][j-1] :
#                 print "element[" + str(i) + "][" + str(j) + "] isn't matching"
#                 print "D_gpu["+str(i)+"]["+str(j)+"] = ", D_gpu[i][j]
#                 print "a[i-1][j-1] =", a[i-1][j-1]
#         elif (i >= M or j >= N) and (i < M+2 and j < N+2) and (D_gpu[i][j] != 0) :
#             print "element[" + str(i) + "][" + str(j) + "] isn't zero!"
#             print "D_gpu["+str(i)+"]["+str(j)+"] = ", D_gpu[i][j]
