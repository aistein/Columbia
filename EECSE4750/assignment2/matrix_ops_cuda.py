#########################################################################################################################
# HEADER
#
# Filename:    matrix_ops_cuda.py
# Description: Implementation of Assignment 2
#
# Owner:       Alexander Stein
# Date:        10/19/2016
# School:      Columbia University
# Class:       EECSE4750
#########################################################################################################################
#########################################################################################################################
# Instructions:
#
#########################################################################################################################
#########################################################################################################################
# References
#
#########################################################################################################################

#########################################################################################################################
# IMPORT STATEMENTS
#########################################################################################################################

# Imports for PyCUDA
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule

# General Imports
import numpy as np
np.set_printoptions(threshold=np.nan)
import sys
import string
import random
import math
import time

#########################################################################################################################
# CONFIGURATIONS
#########################################################################################################################

#########################################################################################################################
# CLASSES
#########################################################################################################################

#########################################################################################################################
# Function Definitions
#########################################################################################################################

def transposePython (A):
    ''' A is expected to be a numpy matrix object of dimensions NxM '''
    return A.T

def transposeCUDA (A):
    ''' A is expected to be a numpy matrix object of dimensionsd M*N '''


    ### 1,2.3. Platform Info, Device Info, and Context are all obtained with the import statement "import pycuda.autoinit"

    ### 4. Create a program for the context, give it a kernel, and build
    mod = SourceModule("""
        __global__ void transpose(const unsigned int *A, unsigned int *T, unsigned int M, unsigned int N)
        {
            // This input will be a 2-D matrix, so we only need three pieces of info to work with it (function of block grid)
    	        unsigned int tidx = threadIdx.x;
    			//unsigned int bidx = blockIdx.x;
    			//unsigned int bdimx = blockDim.x;

            // Transpose: tij = aji = A[j][i] = A_flat[i*N + j]
                unsigned int i, j, aji;
                i = tidx % M;
                j = tidx / M;
                aji = A[i * N + j];

                if(tidx < N * M){
                    T[tidx] = aji;
                }
        }
        """)
    transpose = mod.get_function("transpose")

    ### 5. Command Queue is also handled with import statement "import pycuda.autoinit"

    ### 6. Allocate device memory and move input data from the host to the device memory.
    T = np.empty_like(A)
    M, N = A.shape
    max_size = 1024

    size = A.nbytes

    A_d = cuda.mem_alloc(size)
    T_d = cuda.mem_alloc(size)

    cuda.memcpy_htod(A_d, A)
    cuda.memcpy_htod(T_d, T)

    ### 7. Map buffers to kernel arguments and deploy the kernel, with specified block and grid dimensions
    ###        CUDA organizes memory into a "grid of blocks containing threads."
    ###        Here the grid is 1-D, as are the blocks, each containing 1024 threads.

    b_size = M * N if (M * N <= max_size) else max_size
    g_size = 1 if (M * N <= max_size) else int(math.ceil(M*N/float(max_size)))

    ## Time the deployment of the kernel for metrics
    start = time.time()
    transpose(A_d, T_d, np.uint32(M), np.uint32(N), block=(b_size, 1, 1), grid=(g_size, 1, 1))
    runtime = time.time() - start

    ### 8. Move the kernel's output data back to the host memory
    cuda.memcpy_dtoh(T, T_d)

    return runtime, T

#########################################################################################################################
# Main
#########################################################################################################################

def main(_M_=5, _N_=5):

    '''Default arguments: _M_ - number of columns in input matrix
                          _N_ - number of rows in input matrix'''

    # A = np.matrix('1 2 3 4; 5 6 7 8; 9 10 11 12; 13 14 15 16; 17 18 19 20; 21 22 23 24; 25 26 27 28').astype(np.uint32)
    A = np.matrix(np.random.random_integers(0, 9, (_M_, _N_)).astype(np.uint32))

    start = time.time()
    A_t = transposePython(A)
    runtime = time.time() - start

    print "Matrix: ", A, "\n"
    print "Transpose: ", A_t, "\n"
    print "Flattened: ", A.A1, "\n"
    print runtime

    print "Running CUDA Test\n"
    runtime, T = transposeCUDA(A)
    print np.uint32(T).reshape(A_t.shape)
    print runtime

if __name__ == '__main__':
	main(10, 10)
