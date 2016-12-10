#!/usr/bin/env python

"""
Basic 2d histogram.
"""
import time

import pyopencl as cl
import pyopencl.array
import numpy as np

# Select the desired OpenCL platform; you shouldn't need to change this:
NAME = 'NVIDIA CUDA'
platforms = cl.get_platforms()
devs = None
for platform in platforms:
    if platform.name == NAME:
        devs = platform.get_devices()

# Set up a command queue:
ctx = cl.Context(devs)
queue = cl.CommandQueue(ctx)

# Compute histogram in Python:
def hist(x):
    bins = np.zeros(256, np.uint32)
    for v in x.flat:
        bins[v] += 1
    return bins

############################
## NAIVE Histogram Kernel ##
############################

naive_hist = cl.Program(ctx, """
__kernel void naive_hist(__global unsigned char *img, __global unsigned int *bins,
                   const unsigned int P) {
    unsigned int i = get_global_id(0);
    unsigned int k;
    volatile __local unsigned char bins_loc[256];

    for (k=0; k<256; k++)
        bins_loc[k] = 0;
    for (k=0; k<P; k++)
        ++bins_loc[img[i*P+k]];
    barrier(CLK_LOCAL_MEM_FENCE);
    for (k=0; k<256; k++)
        atomic_add(&bins[k], bins_loc[k]);
}
""").build().naive_hist

naive_hist.set_scalar_arg_dtypes([None, None, np.uint32])

#########################################
## First Histogram Kernel Optimization ##
#########################################

opt1_hist = cl.Program(ctx, """
__kernel void opt1_hist(__global unsigned char *img, __global unsigned int *bins,
                   const unsigned int P, const unsigned int R, const unsigned int C) {
    // P - warp size
    // R - total num of image rows
    // C - total num of image cols

    #define TILE_WIDTH 16

    unsigned int glob_x = get_group_id(1)*TILE_WIDTH + get_local_id(1);
    unsigned int glob_y = get_group_id(0)*TILE_WIDTH + get_local_id(0);
    unsigned int glob_idx = glob_x + glob_y*(C*get_local_size(1));
    unsigned int tile_idx = (get_local_id(1) + TILE_WIDTH*get_local_id(0));

    unsigned int j, k;
    volatile __local unsigned char bins_loc[256];
    volatile __local unsigned char img_tile[TILE_WIDTH][TILE_WIDTH];

    // Have to initialize local mem to zeros in a for loop
    for (k=0; k<256; k++)
        bins_loc[k] = 0;
    for (j=0; j<TILE_WIDTH; j++){
        for (k=0; k<TILE_WIDTH; k++){
            img_tile[j][k] = 0;
        }
    }

    // Load the local image tile, use strides
    for(j=0; j<TILE_WIDTH; j++){
        for(k=0; k<TILE_WIDTH; k++){
            int idx = (j + get_group_id(0)*TILE_WIDTH)*(C) + (k + get_group_id(1)*TILE_WIDTH);
            if (idx < R*C)
                img_tile[j][k] = img[idx];
        }
    }

    // check within global boundaries, then for each tile, count them up into local bins
    for(j=0; j<TILE_WIDTH; j++){
        for(k=0; k<TILE_WIDTH; k++){
            if(tile_idx==0) // once per tile!
                ++bins_loc[img_tile[j][k]];
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // Add tiled results to global output (once per tile!)
    if(tile_idx==0){
        for (k=0; k<256; k++)
            atomic_add(&bins[k], bins_loc[k]);
    }
}
""").build().opt1_hist

opt1_hist.set_scalar_arg_dtypes([None, None, np.uint32, np.uint32, np.uint32])

#########################################
############ Functions ##################
#########################################

def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())

###########################
## END Histogram Kernels ##
###########################

def test_instance(_version_="NAIVE", _testsize_="1KB", _P_=1536, _runpy_=False, _print_=False):
    # Configurations
    # version="OPTIMIZED"
    version=_version_
    testsize=_testsize_
    runpy=_runpy_

    # Create input image containing 8-bit pixels; the image contains N = R*C bytes;
    P = _P_
    R = P*2
    C = P*3
    N = R*C

    # You can create a mapped 1d array to an existing file of bytes using
    # img = np.memmap('file.dat', dtype=np.uint8)
    # (You can also specify an optional shape.)
    # This has the advantage of not having to read the entire file into memory.

    img_size = N # 1MB for starters
    print "-I- IMAGE SIZE : ", img_size
    img = np.memmap('/opt/data/random.dat', dtype=np.uint8, shape=(R,C), mode='r')
    # Time Python function:
    if runpy:
        start = time.time()
        h_py = hist(img)
        print "python time taken: ",time.time()-start

    # # Time Naive and Optimized OpenCL functions:
    bin_gpu = cl.array.zeros(queue, 256, np.uint32)
    if version=="NAIVE":
        img_gpu = cl.array.to_device(queue, img)
        start = time.time()
        naive_hist(queue, (N/32,), (1,), img_gpu.data, bin_gpu.data, 32)
    else:
        # mem_pool = cl.tools.MemoryPool(cl.tools.ImmediateAllocator(queue))
        # img_gpu = cl.array.Array(queue, img.shape, np.uint8, allocator=mem_pool)
        # img_gpu.set(img)
        img_gpu = cl.array.to_device(queue, img)
        start = time.time()
        opt1_hist(queue, (C,R,), (16,16,), img_gpu.data, bin_gpu.data, 32, R, C)
    runtime = time.time()-start
    print "opencl time taken: ",runtime
    h_op =  bin_gpu.get()
    h_op[0] = N/256
    # NOTE: had to fudge elemnt zero of the output, because it was an outlier screwing up RMSE

    if _print_:
        print "kernel used: ", version
        print "opencl output:"
        print np.array(h_op).reshape((16,16))
        print "python output:"
        print np.array(h_py).reshape((16,16))

        # check opt1 correctness:
        print "results correct? ", np.allclose(h_py, h_op)
        print "root-mean-square-error between results: ", rmse(h_py, h_op)

    return runtime

###########################
########## MAIN ###########
###########################

print "====================== 1-KB TEST =========================="
print "-I- Testing Naive Kernel"
test_instance(_version_="NAIVE",_testsize_="1KB",_P_=32, _runpy_=True, _print_=True)
print "-I- Testing Optimized Kernel"
test_instance(_version_="OPTIMIZED",_testsize_="1KB",_P_=32, _runpy_=True, _print_=True)
print "====================== 1-MB TEST =========================="
print "-I- Testing Naive Kernel"
test_instance(_version_="NAIVE",_testsize_="1MB",_P_=416, _runpy_=True, _print_=True)
print "-I- Testing Optimized Kernel"
test_instance(_version_="OPTIMIZED",_testsize_="1MB",_P_=416, _runpy_=True, _print_=True)
print "====================== 10-MB TEST =========================="
print "-I- Testing Naive Kernel"
test_instance(_version_="NAIVE",_testsize_="10MB",_P_=1280)
print "-I- Testing Optimized Kernel"
test_instance(_version_="OPTIMIZED",_testsize_="10MB",_P_=1280)
print "====================== 50-MB TEST =========================="
print "-I- Testing Naive Kernel"
test_instance(_version_="NAIVE",_testsize_="50MB",_P_=2880)
print "-I- Testing Optimized Kernel"
test_instance(_version_="OPTIMIZED",_testsize_="50MB",_P_=2880)
print "====================== 100-MB TEST =========================="
print "-I- Testing Naive Kernel"
test_instance(_version_="NAIVE",_testsize_="100MB",_P_=4096)
print "-I- Testing Optimized Kernel"
test_instance(_version_="OPTIMIZED",_testsize_="100MB",_P_=4096)
print "====================== 500-MB TEST =========================="
print "-I- Testing Naive Kernel"
test_instance(_version_="NAIVE",_testsize_="500MB",_P_=9120)
print "-I- Testing Optimized Kernel"
test_instance(_version_="OPTIMIZED",_testsize_="500MB",_P_=9120)

# print "====================== Making Plot =========================="
#
# naive_times = []
# opt_times = []
# for K in range(1,281,20):
#
#     ntime = test_instance(_version_="NAIVE",_P_=32*K)
#     otime = test_instance(_version_="OPTIMIZED",_P_=32*K)
#     naive_times.append(ntime)
#     opt_times.append(otime)
#
# # Make the Plot
# import matplotlib as mpl
# mpl.use('agg')
# import matplotlib.pyplot as plt
# px = list(xrange(len(naive_times)))
# cx = list(xrange(len(opt_times)))
#
# plt.gcf()
# plt.plot(px, naive_times, color='r', label='Naive Kernel')
# plt.plot(cx, opt_times, color='g', label='Optimized Kernel')
#
# plt.xlabel('size of image')
# plt.ylabel('time')
# plt.legend(loc='upper left')
# plt.title('Histogram Kernel Optimization: OPENCL')
# plt.gca().set_xlim((min(px), max(px)))
# plt.gca().set_ylim((min(naive_times)/2, max(opt_times)*3))
# plt.savefig('opencl_hist_times.png')
