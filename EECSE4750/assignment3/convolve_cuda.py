#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Matrix Convolution in PyCUDA
"""

import numpy as np
# np.set_printoptions(threshold=np.nan)
np.set_printoptions(edgeitems=8)
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

        //// ZERO PADDING - comes before fill so as not to overwrite

        // I (row) interior matrix boundary
        if ( i == M-1 ){
            for(int z=1; z <= F/2; z++){
                DS_A_PAD[(tx+F-z)*(N_LIM+F-1) + (ty+F/2-1)] = 0;
                DS_A_PAD[(tx+F-z)*(N_LIM+F-1) + (ty+F/2)] = 0;
                DS_A_PAD[(tx+F-z)*(N_LIM+F-1) + (ty+F/2+1)] = 0;
            }
        } if ( i == 0 ){
            for(int z=1; z <= F/2; z++){
                DS_A_PAD[(tx+F/2-z)*(N_LIM+F-1) + (ty+F/2-1)] = 0;
                DS_A_PAD[(tx+F/2-z)*(N_LIM+F-1) + (ty+F/2)] = 0;
                DS_A_PAD[(tx+F/2-z)*(N_LIM+F-1) + (ty+F/2+1)] = 0;
            }
        }

        // J (col) interior matrix boundary
        if ( j == N-1 ){
            for(int z=1; z <= F/2; z++){
                DS_A_PAD[(tx+F/2-1)*(N_LIM+F-1) + (ty+F-z)] = 0;
                DS_A_PAD[(tx+F/2)*(N_LIM+F-1) + (ty+F-z)] = 0;
                DS_A_PAD[(tx+F/2+1)*(N_LIM+F-1) + (ty+F-z)] = 0;
            }
        } if ( j == 0 ){
            for(int z=1; z <=F/2; z++){
                DS_A_PAD[(tx+F/2-1)*(N_LIM+F-1) + (ty+F/2-z)] = 0;
                DS_A_PAD[(tx+F/2)*(N_LIM+F-1) + (ty+F/2-z)] = 0;
                DS_A_PAD[(tx+F/2+1)*(N_LIM+F-1) + (ty+F/2-z)] = 0;
            }
        }

        // Matrix Corner boundaries
        if (i == 0 && j == 0){
        // upper left
            for(int q=1; q <= F/2; q++){
                for(int z=0; z <= F/2; z++){
                    DS_A_PAD[z*(N_LIM+F-1)+q] = 0;
                }
            }
        }
        else if (i == M-1 && j == 0){
        // lower left
            for(int q=1; q <= F/2; q++){
                for(int z=0; z <= F/2; z++){
                    DS_A_PAD[(tx+F/2-z)*(N_LIM+F-1)+q] = 0;
                }
            }
        }
        else if (i == 0 && j == N-1){
        // upper right
            for(int q=1; q <= F/2; q++){
                for(int z=0; z <= F/2; z++){
                    DS_A_PAD[z*(N_LIM+F-1) + (ty+q)] = 0;
                }
            }
        }
        else if (i == M-1 && j == N-1){
        // lower right
            for(int q=1; q <= F/2; q++){
                for(int z=0; z <= F/2; z++){
                    DS_A_PAD[(tx+z+1)*(N_LIM+F-1) + (ty+q+1)] = 0;
                }
            }
        }

        __syncthreads();

        //// FILL VALUES OF DS_A_PAD

        bool left = ( ty == 0 ) ? true : false;
        bool right = ( ty == N_LIM-1 ) ? true : false;
        bool top = ( tx == 0 ) ? true : false;
        bool bottom = ( tx == M_LIM-1 ) ? true : false;

        if ( left && j != 0 ) {
        // vertical left block edges - CHECK!
            for(int z=1; z <= F/2; z++){
                DS_A_PAD[(tx+F/2)*(N_LIM+F-1)+(ty+F/2-z)] = A[i*N + (j-z)];
            }
        }

        if ( right && j != N-1 ) {
        // veritcal right block edges - WATCH!
            for(int z=1; z <= F/2; z++){
                if ((j+z) < N){
                    DS_A_PAD[(tx+F/2)*(N_LIM+F-1)+(ty+F/2+z)] = A[i*N + (j+z)];
                }
            }
        }

        if ( top && i != 0 ) {
        // horizontal top block edges - CHECK!
            for(int z=1; z <= F/2; z++){
                DS_A_PAD[(tx+F/2-z)*(N_LIM+F-1)+(ty+F/2)] = A[(i-z)*N + j];
            }
        }

        if ( bottom && i != M-1) {
        // horizontal bottom block edges - WATCH!
            for(int z=1; z <= F/2; z++){
                if ((i+z) < M){
                    DS_A_PAD[(tx+F/2+z)*(N_LIM+F-1)+(ty+F/2)] = A[(i+z)*N + j];
                }
            }
        }

        if ( bottom && right && j != N-1 && i != M-1 ) {
        // lower righthand block corners (outside current block) - WATCH!
            for(int q=1; q <= F/2; q++){
                for(int z=1; z <= F/2; z++){
                    if((i+z) < M && (j+q) < N){
                        DS_A_PAD[(tx+F/2+z)*(N_LIM+F-1)+(ty+F/2+q)] = A[(i+z)*N + (j+q)];
                    }
                }
            }
        }

        if ( bottom && left && j != 0 && i < M-1 ) {
        // lower lefthand block corners (outside current block) - WATCH!
            for(int q=1; q <= F/2; q++){
                for(int z=1; z <= F/2; z++){
                    if((i+z) < M && (j+q) >= 0){
                        DS_A_PAD[(tx+F/2+z)*(N_LIM+F-1)+(ty+F/2-q)] = A[(i+z)*N + (j-q)];
                    }
                }
            }
        }

        if ( top && right && i != 0 && j < N-1 ) {
        // upper righthand block corners (outside current block) - WATCH!
            for(int q=1; q <= F/2; q++){
                for(int z=1; z <= F/2; z++){
                    DS_A_PAD[(tx+F/2-z)*(N_LIM+F-1)+(ty+F/2+q)] = A[(i-z)*N + (j+q)];
                }
            }
        }

        if ( top && left && i != 0 && j != 0 ) {
        // upper lefthand block corners (outside current block)
            for(int q=1; q <= F/2; q++){
                for(int z=1; z <= F/2; z++){
                    DS_A_PAD[(tx+F/2-z)*(N_LIM+2)+(ty+F/2-q)] = A[(i-z)*N + (j-q)];
                }
            }
        }

        // internal elements
            DS_A_PAD[(tx+F/2)*(N_LIM+F-1)+(ty+F/2)] = A[i*N + j];

        __syncthreads();

        //// CONVOLUTION Calculation for element (i,j)

        C[i*N + j] = 0;
        for(int m = 0; m < F; m++){
            for(int n = 0; n < F; n++){
                C[i*N + j] += K[m*F + n] * DS_A_PAD[(tx-m+F-1)*(N_LIM+F-1) +(ty-n+F-1)];
            }
        }// end convolution calculation

        __syncthreads();

    }// end output boundary check
}
"""

##################################################
##Python Code starts here
##################################################

## test_instance - runs through a single iteration of the convolution test
def test_instance(M,N,F,a,f,c):

    # get the kernel code from the template
    kernel_code = kernel_code_template

    # compile the kernel code, with options
    TILE_M = str(M) + 'u' if (M <= 32) else str(32) + 'u'
    TILE_N = str(N) + 'u' if (N <= 32) else str(32) + 'u'
    options = "-DTILE_M=" + TILE_M + " -DTILE_N=" + TILE_N
    OPTIONS = [_flag.strip() for _flag in options.split() if _flag.strip()]
    # print OPTIONS
    mod = compiler.SourceModule(kernel_code,options=OPTIONS)

    # get the kernel function from the compiled module
    conv = mod.get_function("convolve2d")

    # create buffers for transfer into GPU
    A_buf = gpuarray.to_gpu(a)
    K_buf = gpuarray.to_gpu(f)
    C_buf = gpuarray.empty((M,N),a.dtype)
    C_gpu = np.empty((M,N),a.dtype)

    # call to conv
    unit_size = np.dtype(np.int32).itemsize
    shared_mem = (M+F-1)*(N+F-1)*unit_size if (M <= 32) and (N <= 32) else (32+F-1)*(32+F-1)*unit_size

    start = time.time()
    conv(A_buf,K_buf,np.int32(M),np.int32(N),np.int32(F),C_buf,block = (32,32,1),grid = (np.int32(M-1/32)+1,np.int32(N-1/32)+1,1),shared=shared_mem)
    gpu_runtime = time.time() - start

    # copy data back from GPU
    C_gpu = C_buf.get()

    return gpu_runtime, C_gpu

##################################################
## MAIN
##################################################

def main(_M_=5, _N_=5, _F_=7):
    # Configurations
    M = _M_ #rows
    N = _N_ #columns
    F = _F_ #square dim of "kernel/filter"
    MAKE_PLOT = True

    a = np.random.randint(0,10, (M,N)).astype(np.int32) #a is matrix which will be convolved

    # create an FxF filter of random numbers
    # f = np.random.randint(-1, 5, (F,F)).astype(np.int32) #f is kernel matrix
    f = np.ones((F,F)).astype(np.int32) #f is kernel matrix
    # f = np.array([[-1., -2. , -1.], [0., 0., 0.], [1., 2., 1.]]).astype(np.int32)

    # mode='same' gives equal (unpadded) input size to OUTPUT size
    # boundary='fill' gives zeros around the input as padding
    start = time.time()
    c = conv2d(a, f, mode='same', boundary='fill')
    runtime = time.time() - start

    print "====================== PART 1 =========================="
    print "--------------------- PYTHON ---------------------------"
    print "a: \n", a
    print "f: \n", f
    print "c: \n", c
    print "c runtime: ", runtime

    print "----------------------- CUDA ----------------------------"
    runtime, c_gpu = test_instance(M,N,F,a,f,c)

    print "c_gpu: \n", c_gpu
    print "c_gpu time: ", runtime
    print "c_gpu == c_cpu ? --> ", np.array_equal(c_gpu,c)

    # print "====================== PART 2 =========================="
    # print "--------------------- PYTHON ---------------------------"
    # M = 3
    # N = 3
    # gpu_times = []
    # cpu_times = []
    # for K in range(1,100):
    #     # scale size of input by K (filter remains same as above, constant)
    #     a = np.random.randint(0,10, (M*K,N*K)).astype(np.int32)
    #
    #     start = time.time()
    #     c = conv2d(a, f, mode='same', boundary='fill')
    #     runtime = time.time() - start
    #     cpu_times.append(runtime)
    #
    #     runtime, c_gpu = test_instance(M*K,N*K,F,a,f,c)
    #     gpu_times.append(runtime)
    #
    # print "cpu_times: \n", cpu_times
    # print "gpu_times: \n", gpu_times
    #
    # if MAKE_PLOT:
    #     import matplotlib as mpl
    #     mpl.use('agg')
    #     import matplotlib.pyplot as plt
    #     px = list(xrange(len(cpu_times)))
    #     cx = list(xrange(len(gpu_times)))
    #     # cox = list(xrange(len(cuda_opt_times)))
    #
    #     plt.gcf()
    #     plt.plot(px, cpu_times, color='r', label='python')
    #     plt.plot(cx, gpu_times, color='g', label='CUDA')
    #     # plt.plot(cox, cuda_opt_times, color='b', label='CUDA Optimized')
    #     plt.xlabel('3x3 image * k, with 3x3 kernel')
    #     plt.ylabel('time')
    #     plt.legend(loc='upper left')
    #     plt.title('Matrix Convolution: Python vs. CUDA')
    #     plt.gca().set_xlim((min(px), max(px)))
    #     plt.gca().set_ylim((min(cpu_times)/2, max(gpu_times)*1.2))
    #     plt.savefig('cuda_scale_image.png')
    #
    # print "====================== PART 3 =========================="
    # print "--------------------- PYTHON ---------------------------"
    # M = 300
    # N = 300
    # gpu_times = []
    # cpu_times = []
    # for K in range(3,32):
    #     # fixed size of input
    #     a = np.random.randint(0,10, (M,N)).astype(np.int32)
    #
    #     # scale size of kernel
    #     F = K
    #     f = np.random.randint(0, 10, (F,F)).astype(np.int32)
    #
    #     start = time.time()
    #     c = conv2d(a, f, mode='same', boundary='fill')
    #     runtime = time.time() - start
    #     cpu_times.append(runtime)
    #
    #     runtime, c_gpu = test_instance(M,N,F,a,f,c)
    #     gpu_times.append(runtime)
    #
    # print "cpu_times: \n", cpu_times
    # print "gpu_times: \n", gpu_times
    #
    # if MAKE_PLOT:
    #     import matplotlib as mpl
    #     mpl.use('agg')
    #     import matplotlib.pyplot as plt
    #     px = list(xrange(len(cpu_times)))
    #     cx = list(xrange(len(gpu_times)))
    #     # cox = list(xrange(len(cuda_opt_times)))
    #
    #     plt.gcf()
    #     plt.plot(px, cpu_times, color='r', label='python')
    #     plt.plot(cx, gpu_times, color='g', label='CUDA')
    #     # plt.plot(cox, cuda_opt_times, color='b', label='CUDA Optimized')
    #     plt.xlabel('3x3 image * k, with 3x3 kernel')
    #     plt.ylabel('time')
    #     plt.legend(loc='upper left')
    #     plt.title('Matrix Convolution: Python vs. CUDA')
    #     plt.gca().set_xlim((min(px), max(px)))
    #     plt.gca().set_ylim((min(cpu_times)/2, max(gpu_times)*1.2))
    #     plt.savefig('cuda_scale_kernel.png')


if __name__ == '__main__':
	main(330,330)
