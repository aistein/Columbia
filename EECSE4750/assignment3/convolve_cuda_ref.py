#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Matrix Multiplication in PyCUDA
"""

import numpy as np
#import PYCUDA modules and libraries
from pycuda import driver, compiler, gpuarray, tools
import sys
#the following module is used to mark the time stamps
import time

# -- initialize the device
import pycuda.autoinit

############################
##CUDA KERNEL
###########################
kernel_code_template = """
//Transpose function
__global__ void matrix_transpose(unsigned int* a, const unsigned int M, const unsigned int N, unsigned int* y) {

    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;

    if(i<N && j<M){
      y[j+M*i] = a[i+N*j];
    }
}

//naive matrix multiplication
__global__ void matrix_basic_mult(const unsigned int L, const unsigned int M, unsigned int *a, unsigned int *b, unsigned int *c) {

   int i,j,k;

   i = blockIdx.y * blockDim.y + threadIdx.y;
   j = blockIdx.x * blockDim.x + threadIdx.x;
   //C[i,j] = (sum over k) a(i,k)*b(k,j)

   if(i<L && j<L) {
       for(k = 0; k<M; k++) {
          c[i*L+j] += a[i*M+k]* b[k*L+j];
       }
   }
}

//using local scalar
__global__ void matrix_local_scalar(const unsigned int L, const unsigned int M, unsigned int *a, unsigned int *b, unsigned int *c) {

   int i,j,k;

   i = blockIdx.y * blockDim.y + threadIdx.y;
   j = blockIdx.x * blockDim.x + threadIdx.x;
   unsigned int temp = 0;   //temporary private variable where the sum will be copied
   //C[i,j] = (sum over k) A(i,k)*B(k,j)
   if(i<L && j<L) {
       for(k = 0; k<M; k++) {
          temp  += a[i*M+k]* b[k*L+j];
       }

       c[i*L+j] = temp; //in the end this temp is assigned to global output
   }
}

//pvt mem - where row of A is copied into the private memory
__global__ void matrix_pvt_mem(const unsigned int L, const unsigned int M, unsigned int *a, unsigned int *b,unsigned int *c) {

   int i,j,k;

   i = blockIdx.y * blockDim.y + threadIdx.y;
   j = blockIdx.x * blockDim.x + threadIdx.x;
   unsigned int temp = 0;
   unsigned int Atemp[1024]; //A will be copied into Atemp which is closer to memory
   //C[i,j] = (sum over k) A(i,k)*B(k,j)

   if(i<L)
      for(k = 0; k<M; k++) Atemp[k] = a[i*M+k]; //row of A being transferred to a private memory array

   if(i<L && j<L) {
      for(k = 0; k<M; k++) {
          temp  += Atemp[k]* b[k*L+j];
      }
      c[i*L+j] = temp;
   }
}
"""

##################################################
##Python Code starts here
##################################################

#run the scheme into loop
x = np.arange(0,300,1)

#define timer arrays - one for OpenCL and one for Python times
transp_py_times = np.zeros_like(x).astype(np.float64) # we want best precision of the timers, hence float64
transp_cu_times = np.zeros_like(x).astype(np.float64)

mult_py_times = np.zeros_like(x).astype(np.float64) # we want best precision of the timers, hence float64
naive_cu_times = np.zeros_like(x).astype(np.float64)
local_cu_times = np.zeros_like(x).astype(np.float64)
pvt_cu_times = np.zeros_like(x).astype(np.float64)

# get the kernel code from the template
kernel_code = kernel_code_template

# compile the kernel code
mod = compiler.SourceModule(kernel_code)

# get the kernel function from the compiled module
transp = mod.get_function("matrix_transpose")
mat_mul_naive = mod.get_function("matrix_basic_mult")
mat_local_sc = mod.get_function("matrix_local_scalar")
mat_pvt_mem = mod.get_function("matrix_pvt_mem")

for r in xrange(1,300,1):

    L = 4*r
    P = L
    M = 3*r

    print '***********part 1 starts here**************'
    a = np.random.randint(0, 9, (L,P)).astype(np.uint32) #a is matrix which will be transposed

    a_buf = gpuarray.to_gpu(a)
    y_buf = gpuarray.empty((P,L),a.dtype)

    y_gpu = np.empty((P,L),a.dtype)
    y_py = np.empty((P,L),a.dtype)

    start = time.time()
    y_py = np.transpose(a)
    transp_py_times[r] = time.time()-start

    ##transpose call - part 1
    start = time.time()
    transp(a_buf,np.uint32(L),np.uint32(P),y_buf,block = (32,32,1),grid = (np.uint32(L/32)+1,np.uint32(P/32)+1,1))
    transp_cu_times[r] = time.time()-start

    y_gpu = y_buf.get()
    print 'a=',a
    print 'and y_py=',y_py
    print ' and y_gpu = ',y_gpu
    print 'py vs gpu transpose equality:   ', np.allclose(y_py,y_gpu)
    print 'symmetric matrices? ', np.allclose(a,y_gpu)
    print 'matrix dimension=',L,'X',P,'   transpose py time:',transp_py_times[r],' transpose cu time:',transp_cu_times[r]
    print '***********part 1 ends here *****************'
    print '                   '
    print '************part 2 starts here **************'

    a = np.random.randint(0, 9, (L,M)).astype(np.uint32) #a is matrix which will be transposed

    a_buf = gpuarray.to_gpu(a)
    y_buf = gpuarray.empty((M,L),a.dtype)
    c1_buf = gpuarray.zeros((L,L),a.dtype)
    c2_buf = gpuarray.zeros((L,L),a.dtype)
    c3_buf = gpuarray.zeros((L,L),a.dtype)

    y_py = np.empty((M,L),a.dtype)
    c1_gpu = np.empty((L,L),a.dtype)
    c2_gpu = np.empty((L,L),a.dtype)
    c3_gpu = np.empty((L,L),a.dtype)
    c_py = np.empty((L,L),a.dtype)

    start = time.time()
    y_py = np.transpose(a)
    c_py = np.dot(a,y_py)
    mult_py_times[r] = time.time()-start

    ##Note: We cannot escape tiling in PyCUDA
    ##Note: One should not pass np.transpose output to the GPU kernel, as numpy transpose does not actually move the data but only changes the representation
    ##Note: In CUDA, the block size need not be divisible factor of output matrix size , unlike OpenCL

    ##Naive call
    start = time.time()
    transp(a_buf,np.uint32(L),np.uint32(M),y_buf,block = (32,32,1),grid = (np.uint32(L/32)+1,np.uint32(P/32)+1,1))
    mat_mul_naive(np.uint32(L),np.uint32(M),a_buf,y_buf,c1_buf,block = (32,32,1),grid = (np.uint32(L/32)+1,np.uint32(P/32)+1,1))
    naive_cu_times[r] = time.time()-start

    ##local scalar call
    start = time.time()
    transp(a_buf,np.uint32(L),np.uint32(M),y_buf,block = (32,32,1),grid = (np.uint32(L/32)+1,np.uint32(P/32)+1,1))
    mat_local_sc(np.uint32(L),np.uint32(M),a_buf,y_buf,c2_buf,block = (32,32,1),grid = (np.uint32(L/32)+1,np.uint32(P/32)+1,1))
    local_cu_times[r] = time.time()-start

    ##private memory call
    start = time.time()
    transp(a_buf,np.uint32(L),np.uint32(M),y_buf,block = (32,32,1),grid = (np.uint32(L/32)+1,np.uint32(P/32)+1,1))
    mat_pvt_mem(np.uint32(L),np.uint32(M),a_buf,y_buf,c3_buf,block = (32,32,1),grid = (np.uint32(L/32)+1,np.uint32(P/32)+1,1))
    pvt_cu_times[r] = time.time()-start

    c1_gpu = c1_buf.get()
    c2_gpu = c2_buf.get()
    c3_gpu = c3_buf.get()

    print 'a=',a
    print ' and y_gpu = ',y_buf.get()
    print ' and py product = ',c_py
    print ' and naive_gpu_prod = ',c1_gpu
    print ' and local_scalar_opt_prod = ',c2_gpu
    print ' and pvt mem scalar opt prod = ',c3_gpu
    print 'matrix product symmetric?   ', np.allclose(c_py,np.transpose(c_py))
#    print 'All matrix products equal ', np.allclose(c_py,c1_gpu)
    print 'All matrix products equal ', (np.allclose(c_py,c1_gpu) and np.allclose(c1_gpu,c2_gpu) and np.allclose(c2_gpu,c3_gpu))
    print 'matrix dimansion=',L,'X',M,'   mult py time:',mult_py_times[r],' gpu naive time:',naive_cu_times[r]
    print ' gpu local scalar time:',local_cu_times[r],' gpu private mem time:',pvt_cu_times[r]
    print '***********part 2 ends here *****************'

# Optional: if you want to plot the function, set MAKE_PLOT to
# True:
MAKE_PLOT = True
if MAKE_PLOT:
    import matplotlib as mpl
    mpl.use('agg')
    import matplotlib.pyplot as plt

    plt.gcf()
    plt.subplot(311)
    plt.plot(x, transp_py_times,'r')
    plt.plot(x, transp_cu_times,'g')
    plt.legend(['python transpose', 'cuda transpose'], loc='upper left')
    plt.xlabel('matrix ratio increase factor')
    plt.ylabel('output coding times')
    plt.gca().set_xlim((min(x), max(x)))

    plt.subplot(312)
    plt.plot(x, mult_py_times,'r')
    plt.plot(x, naive_cu_times,'b')
    plt.plot(x, local_cu_times,'g')
    plt.plot(x, pvt_cu_times,'m')
    plt.legend(['python mult', 'cuda naive mult','cuda local scalar mult','cuda pvt mem mult'], loc='upper left')
    plt.xlabel('matrix ratio increase factor')
    plt.ylabel('output coding times')
    plt.gca().set_xlim((min(x), max(x)))

    plt.subplot(313)
    plt.plot(x, naive_cu_times,'b')
    plt.plot(x, local_cu_times,'g')
    plt.plot(x, pvt_cu_times,'m')
    plt.legend([ 'cuda naive mult','cuda local scalar mult','cuda pvt mem mult'], loc='upper left')
    plt.xlabel('matrix ratio increase factor')
    plt.ylabel('output coding times')
    plt.gca().set_xlim((min(x), max(x)))
   #plt.gcf()
    plt.savefig('transpose_and_mat_mul_cuda_plots.png')
