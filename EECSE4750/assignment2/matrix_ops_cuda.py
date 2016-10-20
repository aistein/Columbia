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

def multiplyPython (A):
    ''' A is expected to be a numpy matrix object of dimension NxM '''
    return np.dot(A, A.T)

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

def naiveMultiplyCUDA (A):
    ''' A is expected to be a numpy matrix object of dimensionsd M*N '''


    ### 1,2.3. Platform Info, Device Info, and Context are all obtained with the import statement "import pycuda.autoinit"

    ### 4. Create a program for the context, give it a kernel, and build
    mod = SourceModule("""
        __global__ void multiply(const unsigned int *A, unsigned int *T, unsigned int *X, unsigned int M, unsigned int N)
        {

                unsigned int tidx = threadIdx.x;
                unsigned int bidx = blockIdx.x;
                unsigned int bdimx = blockDim.x;

                unsigned int id = bidx * bdimx + tidx;

            // Multiplication:
            // X[j*M + i] = Sum[k = 0 --> M]{A[i*M + k] * A_t[k*M + j]}

            // Accumulate Sum of Row-A * Column-T into X[i][j]
                unsigned int i, j, k, xij;
                i = id % M;
                j = id / M;
                xij = 0;
            // Input Matrix 'A' is MxN, and 'T' is NxM, so X is MxM
                if(id < M * M ){
                    for (k = 0; k < N; k++){
                        X[id] += A[j*N + k] * T[k*M + i];
                    }
                }

            // Input Matrix 'A' is MxN, and 'T' is NxM, so X is MxM
                //if(id < M * M){
                //    X[id] = xij;
                //}
        }
        """)
    multiply = mod.get_function("multiply")

    ### 5. Command Queue is also handled with import statement "import pycuda.autoinit"

    ### 6. Allocate device memory and move input data from the host to the device memory.
    T = np.empty_like(A)
    M, N = A.shape
    X = np.matrix(np.zeros((M,M))).astype(np.uint32)
    max_size = 1024

    size = A.nbytes

    A_d = cuda.mem_alloc(size)
    T_d = cuda.mem_alloc(size)
    X_d = cuda.mem_alloc(X.nbytes)

    cuda.memcpy_htod(A_d, A.A1)
    cuda.memcpy_htod(T_d, A.T.A1)
    cuda.memcpy_htod(X_d, X.A1)

    ### 7. Map buffers to kernel arguments and deploy the kernel, with specified block and grid dimensions
    ###        CUDA organizes memory into a "grid of blocks containing threads."
    ###        Here the grid is 1-D, as are the blocks, each containing 1024 threads.

    b_size = M * M if (M * M <= max_size) else max_size
    g_size = 1 if (M * M <= max_size) else int(math.ceil(M*M/float(max_size)))

    ## Time the deployment of the kernel for metrics
    start = time.time()
    multiply(A_d, T_d, X_d, np.uint32(M), np.uint32(N), block=(b_size, 1, 1), grid=(g_size, 1, 1))
    runtime = time.time() - start

    ### 8. Move the kernel's output data back to the host memory
    cuda.memcpy_dtoh(X, X_d)

    return runtime, X

def optimizedMultiplyCUDA (A):
    ''' A is expected to be a numpy matrix object of dimensionsd M*N '''


    ### 1,2.3. Platform Info, Device Info, and Context are all obtained with the import statement "import pycuda.autoinit"

    ### 4. Create a program for the context, give it a kernel, and build
    mod = SourceModule("""
        __global__ void multiply(const unsigned int *A, unsigned int *T, unsigned int *X, unsigned int M, unsigned int N)
        {
            // Define Constants
                #define TILE_WIDTH 2

            // Indices needed for optimization
                unsigned int tx = threadIdx.x;
                unsigned int bx = blockIdx.x;
                unsigned int ty = threadIdx.y;
                unsigned int by = blockIdx.y;

            // Multiplication:
            // X[j*M + i] = Sum[k = 0 --> M]{A[i*M + k] * A_t[k*M + j]}

            // Tiling Optimiziation: Pull tiles of size TILE_WIDTH x TILE_WIDTH into local memory before operation
                __shared__ unsigned int ds_A[TILE_WIDTH][TILE_WIDTH];
                __shared__ unsigned int ds_T[TILE_WIDTH][TILE_WIDTH];

                unsigned int Row = by * blockDim.y + ty;
                unsigned int Col = bx * blockDim.x + tx;
                unsigned int xvalue = 0;

                for(int t = 0; t < (N-1)/TILE_WIDTH + 1; t++){

                    if(Row < M && (t*TILE_WIDTH + tx) < N){
                        ds_A[ty][tx] = A[Row*N + t*TILE_WIDTH + tx];
                    } else {
                        ds_A[ty][tx] = 0;
                    }

                    if( (t*TILE_WIDTH + ty) < N && Col < M){
                        ds_T[ty][tx] = T[(t*TILE_WIDTH + ty)*M + Col];
                    } else {
                        ds_T[ty][tx] = 0;
                    }

                    __syncthreads();

                    for(int i = 0; i < TILE_WIDTH; i++){
                        xvalue += ds_A[ty][i] * ds_T[i][tx];
                    }// multiply and sum the tiled indices

                }// for all tiles

                __syncthreads();

                //X[Row*N + Col] = ds_A[ty][0]*ds_T[0][tx] + ds_A[ty][1]*ds_T[1][tx];
                //if( Row < M && Col < N){
                if ( (Row * M + Col) < M * M){
                    X[Row*M+ Col] = xvalue;
                    //X[Row*M + Col] = Row*M + Col;
                }

        }// end kernel
        """)
    multiply = mod.get_function("multiply")

    ### 5. Command Queue is also handled with import statement "import pycuda.autoinit"

    ### 6. Allocate device memory and move input data from the host to the device memory.
    T = np.empty_like(A)
    M, N = A.shape
    X = np.matrix(np.zeros((M,M))).astype(np.uint32)
    max_size = 1024

    size = A.nbytes

    A_d = cuda.mem_alloc(size)
    T_d = cuda.mem_alloc(size)
    X_d = cuda.mem_alloc(X.nbytes)

    cuda.memcpy_htod(A_d, A.A1)
    cuda.memcpy_htod(T_d, A.T.A1)
    cuda.memcpy_htod(X_d, X.A1)

    ### 7. Map buffers to kernel arguments and deploy the kernel, with specified block and grid dimensions
    ###        CUDA organizes memory into a "grid of blocks containing threads."
    ###        Here the grid is 1-D, as are the blocks, each containing 1024 threads.

    b_size = M * M if (M * M <= max_size) else max_size
    g_size = 1 if (M * M <= max_size) else int(math.ceil(M*M/float(max_size)))

    ## Time the deployment of the kernel for metrics
    start = time.time()
    # multiply(A_d, T_d, X_d, np.uint32(M), np.uint32(N), block=(b_size, 1, 1), grid=(g_size, 1, 1))
    # multiply(A_d, T_d, X_d, np.uint32(M), np.uint32(N), block=(32, 32, 1), grid=(8, 8, 1))
    multiply(A_d, T_d, X_d, np.uint32(M), np.uint32(N), block=(2, 2, 1), grid=(2, 2, 1))
    runtime = time.time() - start

    ### 8. Move the kernel's output data back to the host memory
    cuda.memcpy_dtoh(X, X_d)

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

    TEST_ALL = False

    if (not TEST_ALL):
        # A = np.matrix('1 2 3 4; 5 6 7 8; 9 10 11 12; 13 14 15 16; 17 18 19 20; 21 22 23 24; 25 26 27 28').astype(np.uint32)
        A = np.matrix(np.random.random_integers(0, 10, (_M_, _N_)).astype(np.uint32))
        # A = np.matrix('0 1 2; 3 4 5').astype(np.uint32)
        # A = np.matrix(np.ones((_M_, _N_)).astype(np.uint32))

        start = time.time()
        A_t = transposePython(A)
        runtime = time.time() - start

        print "Matrix: ", A, "\n"
        print "Transpose: ", A_t, "\n"
        print "Flattened: ", A.A1, "\n"

        print "\nRunning Python Test\n"
        print "Python Transpose time: ",runtime

        start = time.time()
        A_m = multiplyPython(A)
        runtime = time.time() - start
        print "A * A_t: \n", A_m, "\n"
        print "is it symmetric? ", isSymmetric(A_m)
        print "Python Multiply time: ",runtime, "\n"

        print "Running OCL Test\n"
        runtime, T = transposeCUDA(A)
        print "Transpose: \n", np.uint32(T).reshape(A_t.shape)
        print "CUDA Transpose time: ",runtime
        runtime, X = naiveMultiplyCUDA(A)
        print "is it symmetric? ", isSymmetric(X)
        print "A * A_t (naive CUDA): \n", np.uint32(X).reshape(_M_, _M_)
        print "CUDA Naive Multiply time: ",runtime
        runtime, Y = optimizedMultiplyCUDA(A)
        print "A * A_t (optimized CUDA): \n", np.uint32(Y).reshape(_M_, _M_)
        print "CUDA Optimized Multiply time: ",runtime

    ### TEST_ALL :
    ###     Collect data on running time it takes to run the cipher in both methods from string length 1
    ###     all the way up to _repetitions_.  Produce a graph if desired
    if TEST_ALL :
        M = 3
        python_times = []
        ocl_times = []
        ocl_opt_times = []

        for k in xrange(1, 50) :

            ### Generate a random uppercase ASCII string of length _repetitions_
            A = np.matrix(np.random.random_integers(0, 10, (k*_M_,k*_N_)).astype(np.uint32))

            python_times_tmp = []
            ocl_times_tmp = []
            ocl_opt_times_tmp = []

            ### Average over M results on the same string to reduce outliers
            for i in xrange(M):

                ### Run the tests, store the results
                pytime, pyout = multiplyPython(A)
                ocltime, oclout = naiveMultiplyOpenCL(A)
                oclotime, ocloout = optimizedMultiplyOpenCL(A)

                python_times_tmp.append(pytime)
                ocl_times_tmp.append(ocltime)
                ocl_opt_times_tmp.append(oclotime)

            python_times.append(np.average(python_times_tmp))
            ocl_times.append(np.average(ocl_times_tmp))
            ocl_opt_times.append(np.average(ocl_opt_times_tmp))

        MAKE_PLOT = True
        if MAKE_PLOT:
            import matplotlib as mpl
            mpl.use('agg')
            import matplotlib.pyplot as plt
            px = list(xrange(len(python_times)))
            ox = list(xrange(len(ocl_times)))
            oox = list(xrange(len(ocl_opt_times)))

            plt.gcf()
            plt.plot(px, python_times, color='r', label='python')
            plt.plot(ox, ocl_times, color='g', label='OpenCL')
            plt.plot(oox, ocl_opt_times, color='b', label='OpenCL Optimized')
            plt.xlabel('(MxN) * k')
            plt.ylabel('time')
            plt.legend(loc='upper left')
            plt.title('Matrix Multiplication: Python vs. OpenCL')
            plt.gca().set_xlim((min(px), max(px)))
            plt.gca().set_ylim((min(python_times)/2, max(ocl_times)*1.2))
            plt.savefig('python_v_ocl_times.png')

if __name__ == '__main__':
	main(4, 5)
