#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Matrix Multiplication in PyOpenCL
"""

import numpy as np
#import PYOPENCL modules and libraries
import pyopencl as cl
import pyopencl.array
import sys
#the following module is used to mark the time stamps
import time


#  Select the desired OpenCL platform; you shouldn't need to change this:
NAME = 'NVIDIA CUDA'
platforms = cl.get_platforms()
devs = None
for platform in platforms:
    if platform.name == NAME:
        devs = platform.get_devices()

# Set up a command queue:
ctx = cl.Context(devs)
queue = cl.CommandQueue(ctx, properties=cl.command_queue_properties.PROFILING_ENABLE)

###########################
#OPENCL KERNEL
###########################
kernel = """

//The following function takes in the MXN matrix and returns the transpose
__kernel void matrix_transpose(__global unsigned int* a, const unsigned int M, const unsigned int N, __global unsigned int* y) {
    // Remember global id = 1 represents coluumn of output and global id 0 represents row of output matrix
    unsigned int i = get_global_id(1);
    unsigned int j = get_global_id(0);

    if(i<M && j<N){ //limit the threads to valid region
      //direct transpose - make column indices row and vice versa
      y[j+M*i] = a[i+N*j];
    }
}

//The following function is the naive implementation - global output += mult(global inputs)
__kernel void matrix_basic_mult(const unsigned int L, const unsigned int M, __global unsigned int *a,__global unsigned int *b, __global unsigned int *c) {

    int i,j,k;

    i = get_global_id(1);
    j = get_global_id(0);

   //C[i,j] = (sum over k) a(i,k)*b(k,j)

   if(i<L && j<L) {
       for(k = 0; k<M; k++) {
          c[i*L+j] += a[i*M+k]* b[k*L+j];
       }
   }
}

//The following function is the implementation where local scalar is used, such that:
//local scalar += mult(global inputs); global output = local scalar
__kernel void matrix_local_scalar(const unsigned int L, const unsigned int M,__global unsigned int *a,__global unsigned int *b, __global unsigned int *c) {

   int i,j,k;

   i = get_group_id(1)*get_local_size(1) + get_local_id(1);
   j = get_group_id(0)*get_local_size(0) + get_local_id(0);

   unsigned int temp = 0;

   //C[i,j]=temp, where temp = (sum over k) A(i,k)*B(k,j)

   if(i<L && j<L) {
       for(k = 0; k<M; k++) {
          temp  += a[i*M+k]* b[k*L+j];
       }

       c[i*L+j] = temp;
   }
}

//The following function stores row of A into the private memory such that:
//temp = mult(Aprivate row, Bglobal)
//c=temp
__kernel void matrix_pvt_mem(const unsigned int L, const unsigned int M, __global unsigned int *a, __global unsigned int *b, __global unsigned int *c) {

   int i,j,k;

   i = get_group_id(1)*get_local_size(1) + get_local_id(1);
   j = get_group_id(0)*get_local_size(0) + get_local_id(0);

   unsigned int temp = 0;
   unsigned int Atemp[1024]; //A will be copied into Atemp which is closer to memory

   //C[i,j] = (sum over k) A(i,k)*B(k,j)

   //copy the row of A into private mem
   if(i<L)
      for(k = 0; k<M; k++) Atemp[k] = a[i*M+k];

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
transp_op_times = np.zeros_like(x).astype(np.float64) #timer to store GPU trnspose time

mult_py_times = np.zeros_like(x).astype(np.float64) #timer to store matrix multiplication times in python
naive_op_times = np.zeros_like(x).astype(np.float64) #timer to store GPU naive implementation time
local_op_times = np.zeros_like(x).astype(np.float64) #timer to store GPU local scalar implementation time
pvt_op_times = np.zeros_like(x).astype(np.float64) #timer to store GPU pvt memory implementation time

# compile the kernel code
prg = cl.Program(ctx,kernel).build()

##start implementing the logic
for r in xrange(1,300,1):

    L = 4*r
    P = L
    M = 3*r

    print '***********part 1 starts here**************'
    a = np.random.randint(0, 9, (L,P)).astype(np.uint32) #a is matrix which will be transposed

    a_buf = cl.array.to_device(queue,a)
    y_buf = cl.array.empty(queue,(P,L),a.dtype) #y is the transposed output

    y_gpu = np.empty((P,L),a.dtype) #ybuf_data to be copied here
    y_py = np.empty((P,L),a.dtype) #python transpose

    #python transpose
    start = time.time()
    y_py = np.transpose(a)
    transp_py_times[r] = time.time()-start

    #openCL transpose
    start = time.time()
    prg.matrix_transpose(queue,(P,L),None,a_buf.data,np.uint32(L),np.uint32(P),y_buf.data)
    transp_op_times[r] = time.time()-start

    y_gpu = y_buf.get()
    print 'a=',a
    print 'and y_py=',y_py
    print ' and y_gpu = ',y_gpu
    print 'py vs gpu transpose equality:   ', np.allclose(y_py,y_gpu)
    print 'symmetric matrices? ', np.allclose(a,y_gpu)
    print 'matrix dimansion=',L,'X',P,'   transpose py time:',transp_py_times[r],' transpose op time:',transp_op_times[r]
    print '***********part 1 ends here *****************'
    print '                   '
    print '************part 2 starts here **************'

    a = np.random.randint(0, 9, (L,M)).astype(np.uint32) #a is matrix which will be transposed

    a_buf = cl.array.to_device(queue,a)
    y_buf = cl.array.empty(queue,(M,L),a.dtype) #transposed output

    #c1 - naive, c2 - local scalar, c3 - pvt mem
    c1_buf = cl.array.zeros(queue,(L,L),a.dtype)
    c2_buf = cl.array.zeros(queue,(L,L),a.dtype)
    c3_buf = cl.array.zeros(queue,(L,L),a.dtype)

    y_py = np.empty((M,L),a.dtype)
    c1_gpu = np.empty((L,L),a.dtype)
    c2_gpu = np.empty((L,L),a.dtype)
    c3_gpu = np.empty((L,L),a.dtype)
    c_py = np.empty((L,L),a.dtype)

    start = time.time()
    y_py = np.transpose(a)
    c_py = np.dot(a,y_py)
    mult_py_times[r] = time.time()-start

    ## please note: We cannot use np.transpose for the input to matrix multiplication because
    ## numpy transpose never actually moves the data, it just changes the representation.
    ##So, when we pass in the data to GPU input, the coreect transpose is not passed

    ##also note that tile size should be factor of minimum global matrix dimensions

    #naive call
    start = time.time()
    prg.matrix_transpose(queue,(M,L),None,a_buf.data,np.uint32(L),np.uint32(M),y_buf.data)
    prg.matrix_basic_mult(queue,(L,L),None,np.uint32(L),np.uint32(M),a_buf.data,y_buf.data,c1_buf.data)
    naive_op_times[r] = time.time()-start

    #local scalar call
    start = time.time()
    prg.matrix_transpose(queue,(M,L),None,a_buf.data,np.uint32(L),np.uint32(M),y_buf.data)
    prg.matrix_local_scalar(queue,(L,L),(4,4),np.uint32(L),np.uint32(M),a_buf.data,y_buf.data,c2_buf.data)
    local_op_times[r] = time.time()-start

    #private memory implementation call
    start = time.time()
    prg.matrix_transpose(queue,(M,L),None,a_buf.data,np.uint32(L),np.uint32(M),y_buf.data)
    prg.matrix_pvt_mem(queue,(L,L),(4,4),np.uint32(L),np.uint32(M),a_buf.data,y_buf.data,c3_buf.data)
    pvt_op_times[r] = time.time()-start

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
    print 'matrix dimansion=',L,'X',M,'   mult py time:',mult_py_times[r],' gpu naive time:',naive_op_times[r]
    print ' gpu local scalar time:',local_op_times[r],' gpu private mem time:',pvt_op_times[r]
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
    plt.plot(x, transp_op_times,'g')
    plt.legend(['python transpose', 'opencl transpose'], loc='upper left')
    plt.xlabel('matrix ratio increase factor')
    plt.ylabel('output coding times')
    plt.gca().set_xlim((min(x), max(x)))

    plt.subplot(312)
    plt.plot(x, mult_py_times,'r')
    plt.plot(x, naive_op_times,'b')
    plt.plot(x, local_op_times,'g')
    plt.plot(x, pvt_op_times,'m')
    plt.legend(['python mult', 'opencl naive mult','opencl local scalar mult','opencl pvt mem mult'], loc='upper left')
    plt.xlabel('matrix ratio increase factor')
    plt.ylabel('output coding times')
    plt.gca().set_xlim((min(x), max(x)))

    plt.subplot(313)
    plt.plot(x, naive_op_times,'b')
    plt.plot(x, local_op_times,'g')
    plt.plot(x, pvt_op_times,'m')
    plt.legend([ 'opencl naive mult','opencl local scalar mult','opencl pvt mem mult'], loc='upper left')
    plt.xlabel('matrix ratio increase factor')
    plt.ylabel('output coding times')
    plt.gca().set_xlim((min(x), max(x)))
   #plt.gcf()
    plt.savefig('transpose_and_mat_mul_opencl_plots.png')
