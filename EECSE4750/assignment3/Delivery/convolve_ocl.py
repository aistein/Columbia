#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Matrix Convolution in PyOPENCL
"""

import numpy as np
# np.set_printoptions(threshold=np.nan)
np.set_printoptions(edgeitems=8)
#import PYOPENCL modules and libraries
import pyopencl as cl
import pyopencl.array
#the following module is used to mark the time stamps
import time
#import necessary scipy libraries
import scipy as sp
import scipy.signal
from scipy.signal import convolve2d as conv2d

#  Select the desired OpenCL platform; you shouldn't need to change this:
NAME = 'NVIDIA CUDA'
platforms = cl.get_platforms()
devs = None
for platform in platforms:
    if platform.name == NAME:
        devs = platform.get_devices()

# Set up a command queue:
cidx = cl.Context(devs)
queue = cl.CommandQueue(cidx, properties=cl.command_queue_properties.PROFILING_ENABLE)

############################
##SAVED CODE
############################

############################
##OPENCL KERNEL
############################

kernel_code_template = """
//2D Convolution function
__kernel void convolve2d(__global int* A, __global int* K,
                           const int M, const int N,
                           const int F, __global int* C)
{

    #ifndef LOCAL_SIZE
        #define LOCAL_SIZE 3u
    #endif

    // A - input size MxN
    // K - Kernel size FxF
    // M - Row Dim of UN-Padded A
    // N - Column Dim of UN-Padded A
    // F - Square Dim of K (Pitch)
    // C - Output size MxN

    int idx = get_local_id(1);
    int idy = get_local_id(0);
    int i = get_group_id(1)*get_local_size(1) + get_local_id(1);
    int j = get_group_id(0)*get_local_size(0) + get_local_id(0);

    if(i < M && j <  N){
    // check to make sure we are within bounds of the overall output size

        // create flattened padded matrix size M+2 x N+2 to use for convolution input
        __local int DS_A_PAD[LOCAL_SIZE];
        int M_LIM = (M > 32) ? ((get_group_id(1)+1)*get_local_size(1) < M) ? get_local_size(1) : get_local_size(1) - (get_group_id(1)+1)*get_local_size(1) % M : M;
        int N_LIM = (N > 32) ? ((get_group_id(0)+1)*get_local_size(0) < N) ? get_local_size(0) : get_local_size(0) - (get_group_id(0)+1)*get_local_size(0) % N : N;

        //// ZERO PADDING - comes before fill so as not to overwrite

        for(int k = 0; k < M_LIM + F-1; k++){
            for(int l = 0; l < N_LIM + F-1; l++){
                DS_A_PAD[k*(N_LIM+F-1) + l] = 0;
            }
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        //// FILL VALUES OF DS_A_PAD

        bool left = ( idy == 0 ) ? true : false;
        bool right = ( idy == N_LIM-1 ) ? true : false;
        bool top = ( idx == 0 ) ? true : false;
        bool bottom = ( idx == M_LIM-1 ) ? true : false;

        if ( left && j != 0 ) {
        // vertical left block edges - CHECK!
            for(int z=1; z <= F/2; z++){
                if ((j-z) < 0){
                    DS_A_PAD[(idx+F/2)*(N_LIM+F-1)+(idy+F/2-z)] = 0;
                } else {
                    DS_A_PAD[(idx+F/2)*(N_LIM+F-1)+(idy+F/2-z)] = A[i*N + (j-z)];
                }
            }
        }

        if ( right && j != N-1 ) {
        // veritcal right block edges - WATCH!
            for(int z=1; z <= F/2; z++){
                if ((j+z) >= N){
                    DS_A_PAD[(idx+F/2)*(N_LIM+F-1)+(idy+F/2+z)] = 0;
                } else {
                    DS_A_PAD[(idx+F/2)*(N_LIM+F-1)+(idy+F/2+z)] = A[i*N + (j+z)];
                }
            }
        }

        if ( top && i != 0 ) {
        // horizontal top block edges - CHECK!
            for(int z=1; z <= F/2; z++){
                if((i-z) < 0){
                    DS_A_PAD[(idx+F/2-z)*(N_LIM+F-1)+(idy+F/2)] = 0;
                } else {
                    DS_A_PAD[(idx+F/2-z)*(N_LIM+F-1)+(idy+F/2)] = A[(i-z)*N + j];
                }
            }
        }

        if ( bottom && i != M-1) {
        // horizontal bottom block edges - WATCH!
            for(int z=1; z <= F/2; z++){
                if ((i+z) >= M){
                    DS_A_PAD[(idx+F/2+z)*(N_LIM+F-1)+(idy+F/2)] = 0;
                } else {
                    DS_A_PAD[(idx+F/2+z)*(N_LIM+F-1)+(idy+F/2)] = A[(i+z)*N + j];
                }
            }
        }

        if ( bottom && right && j != N-1 && i != M-1 ) {
        // lower righthand block corners (outside current block) - WATCH!
            for(int q=1; q <= F/2; q++){
                for(int z=1; z <= F/2; z++){
                    if((i+z) >= M || (j+q) >= N){
                        DS_A_PAD[(idx+F/2+z)*(N_LIM+F-1)+(idy+F/2+q)] = 0;
                    } else {
                        DS_A_PAD[(idx+F/2+z)*(N_LIM+F-1)+(idy+F/2+q)] = A[(i+z)*N + (j+q)];
                    }
                }
            }
        }

        if ( bottom && left && j != 0 && i < M-1 ) {
        // lower lefthand block corners (outside current block) - WATCH!
            for(int q=1; q <= F/2; q++){
                for(int z=1; z <= F/2; z++){
                    if((i+z) >= M || (j+q) < 0){
                        DS_A_PAD[(idx+F/2+z)*(N_LIM+F-1)+(idy+F/2-q)] = 0;
                    } else {
                        DS_A_PAD[(idx+F/2+z)*(N_LIM+F-1)+(idy+F/2-q)] = A[(i+z)*N + (j-q)];
                    }
                }
            }
        }

        if ( top && right && i != 0 && j < N-1 ) {
        // upper righthand block corners (outside current block) - CHECK!
            for(int q=1; q <= F/2; q++){
                for(int z=1; z <= F/2; z++){
                    if((i-z) >= M || (j+q) >= N){
                        DS_A_PAD[(idx+F/2-z)*(N_LIM+F-1)+(idy+F/2+q)] = 0;
                    } else {
                        DS_A_PAD[(idx+F/2-z)*(N_LIM+F-1)+(idy+F/2+q)] = A[(i-z)*N + (j+q)];
                    }
                }
            }
        }

        if ( top && left && i != 0 && j != 0 ) {
        // upper lefthand block corners (outside current block)
            for(int q=1; q <= F/2; q++){
                for(int z=1; z <= F/2; z++){
                    if((i-z) < 0 || (j-q) <0){
                        DS_A_PAD[(idx+F/2-z)*(N_LIM+F-1)+(idy+F/2-q)] = 0;
                    } else {
                        DS_A_PAD[(idx+F/2-z)*(N_LIM+F-1)+(idy+F/2-q)] = A[(i-z)*N + (j-q)];
                    }
                }
            }
        }

        // internal elements
            DS_A_PAD[(idx+F/2)*(N_LIM+F-1)+(idy+F/2)] = A[i*N + j];

        barrier(CLK_LOCAL_MEM_FENCE);

        C[i*N+j] = DS_A_PAD[(idx+F/2)*(N_LIM+F-1)+(idy+F/2)];

        //// CONVOLUTION Calculation for element (i,j)

        C[i*N + j] = 0;
        for(int m = 0; m < F; m++){
            for(int n = 0; n < F; n++){
                C[i*N + j] += K[m*F + n] * DS_A_PAD[(idx-m+F-1)*(N_LIM+F-1) +(idy-n+F-1)];
            }
        }// end convolution calculation

        barrier(CLK_LOCAL_MEM_FENCE);

    }// end output boundary check

}
"""

##################################################
##Python Code starts here
##################################################

## test_instance - runs through a single iteration of the convolution test
def test_instance(M,N,F,a,f,c):

    # set build options / local memory size
    LSIZE = str((32+F-1)*(32+F-1)) + 'u'
    options = "-DLOCAL_SIZE=" + LSIZE

    # build the kernel
    prg = cl.Program(cidx,kernel_code_template).build(options.strip())

    # get the kernel code from the template
    kernel_code = kernel_code_template

    # create buffers for transfer into GPU
    A_buf = cl.array.to_device(queue,a)
    K_buf = cl.array.to_device(queue,f)
    C_buf = cl.array.empty(queue,(M,N),a.dtype)
    C_gpu = np.empty((M,N),a.dtype)

    # dimensions to deploy
    global_dim = ((np.int32(M-1/32)+1)*32,(np.int32(N-1/32)+1)*32,1)
    local_dim = (32,32,1)

    # deploy kernel
    start = time.time()
    prg.convolve2d(queue,global_dim,local_dim,A_buf.data,K_buf.data,np.int32(M),np.int32(N),np.int32(F),C_buf.data)
    gpu_runtime = time.time() - start

    # copy data back from GPU
    C_gpu = C_buf.get()

    return gpu_runtime, C_gpu

##################################################
## MAIN
##################################################

def main(_M_=5, _N_=5, _F_=3):

    print "================== CONFIGURATIONS ========================"
    M = _M_ #rows
    N = _N_ #columns
    F = _F_ #square dim of "kernel/filter"

    MAKE_PLOT_PART2 = False
    MAKE_PLOT_PART3 = False

    print "M: ", M
    print "N: ", N
    print "Kernel: ", F

    print "====================== PART 1-A =========================="
    print "--------------------- PYTHON ---------------------------"

    a = np.random.randint(0,10, (M,N)).astype(np.int32) #a is matrix which will be convolved

    # create an FxF filter of random numbers
    # f = np.random.randint(0, 5, (F,F)).astype(np.int32) #f is kernel matrix
    # f = np.ones((F,F)).astype(np.int32) #f is kernel matrix
    f = np.array([[-1., -2. , -1.], [0., 0., 0.], [1., 2., 1.]]).astype(np.int32)

    # mode='same' gives equal (unpadded) input size to OUTPUT size
    # boundary='fill' gives zeros around the input as padding
    start = time.time()
    c = conv2d(a, f, mode='same', boundary='fill')
    runtime = time.time() - start

    print "a: \n", a
    print "f: \n", f
    print "c: \n", c
    print "c runtime: ", runtime

    print "----------------------- OPENCL ----------------------------"
    runtime, c_gpu = test_instance(M,N,F,a,f,c)

    print "c_gpu: \n", c_gpu
    print "c_gpu time: ", runtime
    print "c_gpu == c_cpu ? --> ", np.allclose(c_gpu,c)

    print "====================== PART 1-B =========================="
    print "--------------------- PYTHON ---------------------------"
    M = 3
    N = 3
    gpu_times = []
    gpu_results = []
    cpu_times = []
    cpu_results = []
    for K in range(1,100):
        # scale size of input by K (filter remains same as above, constant)
        a = np.random.randint(0,10, (M*K,N*K)).astype(np.int32)

        start = time.time()
        c = conv2d(a, f, mode='same', boundary='fill')
        runtime = time.time() - start
        cpu_times.append(runtime)
        cpu_results.append(c)

        runtime, c_gpu = test_instance(M*K,N*K,F,a,f,c)
        gpu_times.append(runtime)
        gpu_results.append(c_gpu)

    # print "cpu_times: \n", cpu_times
    # print "gpu_times: \n", gpu_times

    flag = True
    for x in range(len(gpu_results)):
        if(not np.allclose(gpu_results[x],cpu_results[x])):
            flag = False
    print "All GPU results == All CPU results? --> ", flag

    if MAKE_PLOT_PART2:
        import matplotlib as mpl
        mpl.use('agg')
        import matplotlib.pyplot as plt
        px = list(xrange(len(cpu_times)))
        cx = list(xrange(len(gpu_times)))
        # cox = list(xrange(len(opencl_opt_times)))

        plt.gcf()
        plt.plot(px, cpu_times, color='r', label='python')
        plt.plot(cx, gpu_times, color='g', label='OPENCL')
        # plt.plot(cox, opencl_opt_times, color='b', label='OPENCL Optimized')
        plt.xlabel('3x3 image * k, with 3x3 kernel')
        plt.ylabel('time')
        plt.legend(loc='upper left')
        plt.title('Matrix Convolution: Python vs. OPENCL')
        plt.gca().set_xlim((min(px), max(px)))
        plt.gca().set_ylim((min(cpu_times)/2, max(gpu_times)*1.2))
        plt.savefig('opencl_scale_image.png')

    print "====================== PART 1-C =========================="
    print "--------------------- PYTHON ---------------------------"
    M = 300
    N = 300
    gpu_times = []
    gpu_results = []
    cpu_times = []
    cpu_results = []
    for K in range(3,32,2):
        # fixed size of input
        a = np.random.randint(0,10, (M,N)).astype(np.int32)

        # scale size of kernel
        F = K
        f = np.random.randint(0, 10, (F,F)).astype(np.int32)

        start = time.time()
        c = conv2d(a, f, mode='same', boundary='fill')
        runtime = time.time() - start
        cpu_times.append(runtime)
        cpu_results.append(c)

        runtime, c_gpu = test_instance(M,N,F,a,f,c)
        gpu_times.append(runtime)
        gpu_results.append(c_gpu)

    # print "cpu_times: \n", cpu_times
    # print "gpu_times: \n", gpu_times

    flag = True
    for x in range(len(gpu_results)):
        if(not np.allclose(gpu_results[x],cpu_results[x])):
            flag = False
    print "All GPU results == All CPU results? --> ", flag

    if MAKE_PLOT_PART3:
        import matplotlib as mpl
        mpl.use('agg')
        import matplotlib.pyplot as plt
        px = list(xrange(len(cpu_times)))
        cx = list(xrange(len(gpu_times)))
        # cox = list(xrange(len(opencl_opt_times)))

        plt.gcf()
        plt.plot(px, cpu_times, color='r', label='python')
        plt.plot(cx, gpu_times, color='g', label='OPENCL')
        # plt.plot(cox, opencl_opt_times, color='b', label='OPENCL Optimized')
        plt.xlabel('300x200 image, with 3kx3k kernel')
        plt.ylabel('time')
        plt.legend(loc='upper left')
        plt.title('Matrix Convolution: Python vs. OPENCL')
        plt.gca().set_xlim((min(px), max(px)))
        plt.gca().set_ylim((min(gpu_times)/2, max(gpu_times)*40))
        plt.savefig('opencl_scale_kernel.png')

    print "====================== PART 2 =========================="
    print "--------------------- PYTHON ---------------------------"

    from PIL import Image
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    ## Use the function below to dynamically change image size.
    def create_img(filename, cols , rows):
            size = (cols,rows)
            im = Image.open(filename).convert('L') #.convert('L') converts the image to grayscale
            im = im.resize(size)
            return np.array(im).astype(np.int32)

    ## To download the image:
    ## wget http://heroichollywood.com/wp-content/uploads/2016/06/thrones-002-2.jpg
    ##
    ## Partially based on:
    ## http://www.programcreek.com/python/example/58254/scipy.signal.convolve2d

    original_image = create_img("./thrones-002-2.jpg", 612,380)

    big_smooth = np.array(
    [[0., 1., 2., 4., 8., 16., 8., 4., 2., 1., 0.],
     [1., 2., 4., 8., 16., 32., 16., 8., 4., 2., 1.],
     [2., 4., 8., 16., 32., 64., 32., 16., 8., 4., 2.],
     [4., 8., 16., 32., 64., 128., 64., 32., 16., 8., 4.],
     [8., 16., 32., 64., 128., 256., 128., 64., 32., 16., 8.],
     [16., 32., 64., 128., 256., 512., 256., 128., 64., 32., 16.],
     [8., 16., 32., 64., 128., 256., 128., 64., 32., 16., 8.],
     [4., 8., 16., 32., 64., 128., 64., 32., 16., 8., 4.],
     [2., 4., 8., 16., 32., 64., 32., 16., 8., 4., 2.],
     [1., 2., 4., 8., 16., 32., 16., 8., 4., 2., 1.],
     [0., 1., 2., 4., 8., 16., 8., 4., 2., 1., 0.],
     ]).astype(np.int32)

    filters = {
                    'identity':np.array([ [0.,0.,0.],[0.,1.,0.],[0.,0.,0.]  ]).astype(np.int32),
                    'sharpen':np.array([[0., -1. , 0.], [-1., 5., -1.], [0., -1., 0]]).astype(np.int32),
                    'blur':np.array([[1., 1. , 1.], [1., 1., 1.], [1., 1., 1.]]).astype(np.int32),
                    'edge_det':np.array([[0., 1. , 0.], [1., -4., 1.], [0., 1., 0]]).astype(np.int32),
                    'emboss':np.array([[2., 1. , 0.], [1., 1., -1.], [0., -1., -2]]).astype(np.int32),
                    'sob_x':np.array([[-1., 0. , 1.], [-2., 0., 2.], [-1., 0., 1]]).astype(np.int32),
                    'sob_y':np.array([[-1., -2. , -1.], [0., 0., 0.], [1., 2., 1.]]).astype(np.int32),
                    'smooth_5x5':np.array([[0., 1., 2. , 1., 0.], [1., 4., 8., 4., 1.],[2.,8.,16.,8.,2.],[1.,4.,8.,4.,1.], [0.,1., 2., 1.,0.]]).astype(np.int32),
                    'smooth_11x11':big_smooth
            }

    for filter in filters:
        runtime, c_gpu = test_instance(380,612,len(filters[filter][0]),original_image.astype(np.int32),filters[filter],c)
        c_cpu = conv2d(original_image.astype(np.int32),filters[filter], mode='same', boundary='fill')
        gpu_img = Image.fromarray(c_gpu.astype(np.uint8))
        cpu_img = Image.fromarray(c_cpu.astype(np.uint8))
        cpu_fn = str(filter) + '_img_cpu_opencl.png'
        gpu_fn = str(filter) + '_img_gpu_opencl.png'
        cpu_img.save(cpu_fn)
        gpu_img.save(gpu_fn)
        print "Filter: " + filter
        print "c_gpu == c_cpu ? --> ", np.allclose(c_gpu,c_cpu)

if __name__ == '__main__':
	main(640,640)
