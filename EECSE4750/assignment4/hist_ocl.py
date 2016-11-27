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

# Create input image containing 8-bit pixels; the image contains N = R*C bytes;
# P = 32
# R = P*2
# C = P*3
# N = R*C

P = 512
R = P*2
C = P*3
N = R*C
# img = np.random.randint(0, 255, N).astype(np.uint8).reshape(R, C)

# You can create a mapped 1d array to an existing file of bytes using
# img = np.memmap('file.dat', dtype=np.uint8)
# (You can also specify an optional shape.)
# This has the advantage of not having to read the entire file into memory.

img_size = N # 1MB for starters
print "-I- IMAGE SIZE : ", img_size
img = np.memmap('/opt/data/random.dat', dtype=np.uint8, shape=(R,C), mode='r')

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
""").build().opt1_hist

opt1_hist.set_scalar_arg_dtypes([None, None, np.uint32])

###########################
## END Histogram Kernels ##
###########################

# Time Python function:
start = time.time()
h_py = hist(img)
print time.time()-start

# # Time Naive OpenCL function:
# img_gpu = cl.array.to_device(queue, img)
# bin_gpu = cl.array.zeros(queue, 256, np.uint32)
# start = time.time()
# naive_hist(queue, (N/32,), (1,), img_gpu.data, bin_gpu.data, 32)
# print time.time()-start
# h_op =  bin_gpu.get()
#
# # Check naive correctness:
# print np.allclose(h_py, h_op)

# Time Opt1 OpenCL function: ALLOC_HOST_PTR
# OPT1: using memory pool to keep memory allocated
# OPT2: using strides to coalesce dram accesses
# OPT3: change order from 'C' column major to 'R' row major
mem_pool = cl.tools.MemoryPool(cl.tools.ImmediateAllocator(queue))
img_gpu_opt = cl.array.Array(queue, img, allocator=mem_pool, strides=(32,), order='R')
img_gpu_opt.to_device()
bin_gpu_opt = cl.array.zeros(queue, 256, np.uint32)
start = time.time()
opt1_hist(queue, (N/32,), (1,), img_gpu_opt.data, bin_gpu_opt.data, 32)
print time.time()-start
h_op_opt =  bin_gpu_opt.get()

# check opt1 correctness:
print np.allclose(h_py, h_op_opt)
