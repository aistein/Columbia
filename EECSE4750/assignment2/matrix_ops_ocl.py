#########################################################################################################################
# HEADER
#
# Filename:    matrix_ops_ocl.py
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
# https://github.com/pyopencl/pyopencl/blob/master/examples/transpose.py - Simple pyOpenCL Transpose example
#
#########################################################################################################################

#########################################################################################################################
# IMPORT STATEMENTS
#########################################################################################################################

# Imports for PyOpenCL
import pyopencl as cl
from pyopencl import array

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

def multiplyPython (A):
    ''' A is expected to be a numpy matrix object of dimension NxM '''
    return np.dot(A, A.T)

def transposeOpenCL (A):
    ''' A is expected to be a numpy matrix object of dimensionsd M*N '''

    ### 1. Obtain OpenCL Platform
    platform = cl.get_platforms()[0]

    ### 2. Obtain Device ID for GPU
    device_id = platform.get_devices()[0]

    ### 3. Create Context for selected device
    context = cl.Context([device_id])

    ### 4. Create a program for the context, give it a kernel, and build
    program = cl.Program(context,
        """
    	__kernel void transpose(__global const unsigned int *A, __global unsigned int *T, unsigned int M, unsigned int N)
    	{

            unsigned int idx = get_global_id(0);
            unsigned int idy = get_global_id(1);
            unsigned int dimx = get_global_size(0);

            unsigned int id = dimx * idy + idx;


            // Transpose: tij = aji = A[j][i] = A_flat[i*N + j]
            unsigned int i, j, aji;
            i = idx % M;
            j = idx / M;
            aji = A[i * N + j];

            if(idx < N * M){
                T[idx] = aji;
            }

    	}// end kernel
    	""").build()

    ### 5. Create a command queue for the target device
    queue = cl.CommandQueue(context)

    ### 6. Allocate device memory and move input data from the host to the device memory.
    mem_flags = cl.mem_flags
    M, N = A.shape
    max_size = 1024

    # N_buf = cl.Buffer(context, mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR, size=N.nbytes, hostbuf=N)
    A_buf = cl.Buffer(context, mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR, hostbuf=A.A1)
    T = np.empty_like(A.A1)
    T_buf = cl.Buffer(context, mem_flags.WRITE_ONLY, A.A1.nbytes)

    ### 7. Map buffers to kernel arguments and deploy the kernel, with specified local and global dimensions
    ###    Choosing Dimensions:
    ###        A given work-group cannot exceed 1024 work-items; if > 1024 are required, A new rank is created
    ###        Excess work-item space is trimmed in the kernel with 'if ((gdim_x * gid_y + gid_x) < len){''

    global_dim = (max_size, int(math.ceil(M*N/float(max_size))))
    # print global_dim
    local_dim = (max_size,1)
    # print local_dim

    ## Time the deployment of the kernel for metrics
    start = time.time()
    program.transpose(queue, global_dim, local_dim, A_buf, T_buf, np.uint32(M), np.uint32(N))
    runtime = time.time() - start

    ### 8. Move the kernel's output data back to the host memory
    cl.enqueue_copy(queue, T, T_buf)

    return runtime, T

def naiveMultiplyOpenCL (A):
    ''' A is expected to be a numpy matrix object of dimensionsd M*N '''

    ### 1. Obtain OpenCL Platform
    platform = cl.get_platforms()[0]

    ### 2. Obtain Device ID for GPU
    device_id = platform.get_devices()[0]

    ### 3. Create Context for selected device
    context = cl.Context([device_id])

    ### 4. Create a program for the context, give it a kernel, and build
    program = cl.Program(context,
        """
    	__kernel void transpose(__global const unsigned int *A, __global unsigned int *T, __global unsigned int *X, unsigned int M, unsigned int N)
    	{

                unsigned int idx = get_global_id(0);
                unsigned int idy = get_global_id(1);
                unsigned int dimx = get_global_size(0);

                unsigned int id = dimx * idy + idx;

            // Multiplication:
            // X[j*M + i] = Sum[k = 0 --> M]{A[i*M + k] * A_t[k*M + j]}

            // Accumulate Sum of Row-A * Column-T into X[i][j]
                unsigned int i, j, k, xij;
                i = id % M;
                j = id / M;
                xij = 0;
                for (k = 0; k < N; k++){
                    xij += A[j*N + k] * T[k*M + i];
                }

            // Input Matrix 'A' is MxN, and 'T' is NxM, so X is MxM
                if(id < M * M){
                    X[id] = xij;
                }

    	}// end kernel
    	""").build()

    ### 5. Create a command queue for the target device
    queue = cl.CommandQueue(context)

    ### 6. Allocate device memory and move input data from the host to the device memory.
    mem_flags = cl.mem_flags
    M, N = A.shape
    max_size = 1024

    ### Note: *.A1 flattens the matrix into a row-major 1-D array
    A_buf = cl.Buffer(context, mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR, hostbuf=A.A1)
    T_buf = cl.Buffer(context, mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR, hostbuf=A.T.A1)
    X = np.empty_like(np.matrix(np.zeros((M,M)))).astype(np.uint32)
    X_buf = cl.Buffer(context, mem_flags.WRITE_ONLY, X.nbytes)

    ### 7. Map buffers to kernel arguments and deploy the kernel, with specified local and global dimensions
    ###    Choosing Dimensions:
    ###        A given work-group cannot exceed 1024 work-items; if > 1024 are required, A new rank is created
    ###        Excess work-item space is trimmed in the kernel with 'if ((gdim_x * gid_y + gid_x) < len){''

    global_dim = (max_size, int(math.ceil(M*N/float(max_size))))
    # print global_dim
    local_dim = (max_size,1)
    # print local_dim

    ## Time the deployment of the kernel for metrics
    start = time.time()
    program.transpose(queue, global_dim, local_dim, A_buf, T_buf, X_buf, np.uint32(M), np.uint32(N))
    runtime = time.time() - start

    ### 8. Move the kernel's output data back to the host memory
    cl.enqueue_copy(queue, X, X_buf)

    return runtime, X

def isSymmetric(A):
    ''' expects a numpy matrix of dimensions NxM '''
    return (A.T == A).all()

#########################################################################################################################
# Main
#########################################################################################################################

def main(_M_=5, _N_=5):

    '''Default arguments: _M_ - number of columns in input matrix
                          _N_ - number of rows in input matrix'''

    # A = np.matrix('1 2 3 4; 5 6 7 8; 9 10 11 12; 13 14 15 16; 17 18 19 20; 21 22 23 24; 25 26 27 28').astype(np.uint32)
    A = np.matrix(np.random.random_integers(0, 10, (_M_, _N_)).astype(np.uint32))
    # A = np.matrix(np.ones((_M_, _N_)).astype(np.uint32))

    start = time.time()
    A_t = transposePython(A)
    runtime = time.time() - start

    # print "Matrix: ", A, "\n"
    # print "Transpose: ", A_t, "\n"
    # print "Flattened: ", A.A1, "\n"

    print "\nRunning Python Test\n"
    print "Python Transpose time: ",runtime

    start = time.time()
    A_m = multiplyPython(A)
    runtime = time.time() - start
    # print "A * A_t: \n", A_m, "\n"
    print "is it symmetric? ", isSymmetric(A_m)
    print "Python Multiply time: ",runtime, "\n"

    print "Running OCL Test\n"
    runtime, T = transposeOpenCL(A)
    # print "Transpose: \n", np.uint32(T).reshape(A_t.shape)
    print "OCL Transpose time: ",runtime
    runtime, X = naiveMultiplyOpenCL(A)
    print "is it symmetric? ", isSymmetric(X)
    # print "A * A_t (naive OCL): \n", np.uint32(X).reshape(_M_, _M_)
    print "OCL Naive Multiply time: ",runtime

if __name__ == '__main__':
	main(100, 100)
