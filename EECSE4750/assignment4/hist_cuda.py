#!/usr/bin/env python

"""
Basic 2d histogram.
"""
import time

from pycuda import driver, compiler, gpuarray, tools
import pycuda.autoinit
import numpy as np

# Compute histogram in Python:
def hist(x):
    bins = np.zeros(256, np.uint32)
    for v in x.flat:
        bins[v] += 1
    return bins

# Create input image containing 8-bit pixels; the image contains N = R*C bytes;
P = 32
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
# print img.flat[6143]

############################
## NAIVE Histogram Kernel ##
############################

kernel_code_template="""
// NAIVE KERNEL: unoptimized
__global__ void naive_hist(unsigned char *img, unsigned int *bins,
                   const unsigned int P) {
    unsigned int i = blockIdx.x;
    unsigned int k;
    volatile __shared__ unsigned char bins_loc[256];

    for (k=0; k<256; k++)
        bins_loc[k] = 0;
    for (k=0; k<P; k++)
        ++bins_loc[img[i*P+k]];
    __syncthreads();
    for (k=0; k<256; k++)
        atomicAdd(&bins[k], bins_loc[k]);
}

// OPTIMIZED KERNEL: tiling
__global__ void opt_hist(unsigned char*img, unsigned int *bins,
                        const unsigned int P, const unsigned int R, const unsigned int C) {

    // P - warp size (32)
    // R - total num of image rows
    // C - total num of image cols

    #define TILE_WIDTH 16

    unsigned int glob_x = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int glob_y = blockIdx.y*blockDim.y + threadIdx.y;
    unsigned int glob_idx = glob_x + glob_y*(gridDim.x*blockDim.x);
    unsigned int tile_idx = (threadIdx.x + TILE_WIDTH*threadIdx.y);

    unsigned int j, k;
    volatile __shared__ unsigned char bins_loc[256];
    volatile __shared__ unsigned char img_tile[TILE_WIDTH][TILE_WIDTH];

    // Have to initialize local mem to zeros in a for loop
    for (k=0; k<256; k++)
        bins_loc[k] = 0;
    for (j=0; j<TILE_WIDTH; j++){
        for (k=0; k<TILE_WIDTH; k++){
            img_tile[j][k] = 0;
        }
    }

    // Load the local image tile
    //j = threadIdx.y;
    //k = threadIdx.x;
    //if(glob_x < C && glob_y < R){
    //        // desc:[loc within tile ]       (            tile_y + global-y-offset            )   ( tile_x + global-x-offset )
    //        img_tile[j][k] = img[(j + blockIdx.y*blockDim.y)*(gridDim.x*blockDim.x) + (k + blockIdx.x*blockDim.x)];
    //}
    //__syncthreads();
    for(j=0; j<TILE_WIDTH; j++){
        for(k=0; k<TILE_WIDTH; k++){
            // desc:[loc within tile ]       (            tile_y + global-y-offset            )   ( tile_x + global-x-offset )
            img_tile[j][k] = img[(j + blockIdx.y*blockDim.y)*(gridDim.x*blockDim.x) + (k + blockIdx.x*blockDim.x)];
        }
    }

    // check within global boundaries, then for each tile, count them up into local bins
    //if(glob_x < C && glob_y < R)
    //    ++bins_loc[img_tile[threadIdx.y][threadIdx.x]];
    //__syncthreads();
    for(j=0; j<TILE_WIDTH; j++){
        for(k=0; k<TILE_WIDTH; k++){
            if(tile_idx==0) // once per tile!
                ++bins_loc[img_tile[j][k]];
        }
    }
    __syncthreads();

    // Add tiled results to global output (once per tile!)
    if(tile_idx==0){
        for (k=0; k<256; k++)
            atomicAdd(&bins[k], bins_loc[k]);
    }
}
"""

#########################################
############ Functions ##################
#########################################

def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())

###########################
## END Histogram Kernels ##
###########################

# Configurations
version="NAIVE"

# Time Python function:
start = time.time()
h_py = hist(img)
print time.time()-start

# Time Naive OpenCL function:

# build cuda kernel
kernel_code = kernel_code_template
mod = compiler.SourceModule(kernel_code)

# get functions from kernel_code
naive_hist = mod.get_function("naive_hist")
opt_hist = mod.get_function("opt_hist")

# create memory buffers
img_gpu = gpuarray.to_gpu(img)
bin_gpu = gpuarray.zeros(256, np.uint32)

# allocate shared memory
# unit_size = np.dtype(np.uint32).itemsize
# shared_mem = unit_size*(np.uint32(256))

# deploy naive kernel
start = time.time()
if version=="NAIVE":
    # naive_hist(img_gpu.data, bin_gpu.data, np.uint32(32), block=(1,), grid=(N/32,), shared=shared_mem)
    naive_hist(img_gpu, bin_gpu, np.uint32(32), block=(1,1,1), grid=(N/32,1,1))
else:
    opt_hist(img_gpu, bin_gpu, np.uint32(P), np.uint32(R), np.uint32(C), block=(16,16,1), grid=(C/16,R/16,1))
print time.time()-start
h_op =  bin_gpu.get()

# Check naive correctness:
print np.allclose(h_py, h_op)
print rmse(h_py, h_op)
print "PY: ", h_py
print "CU: ", h_op
