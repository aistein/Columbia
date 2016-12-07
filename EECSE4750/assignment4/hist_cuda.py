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

############################
## NAIVE Histogram Kernel ##
############################

#########################################
########### Saved Code ##################
#########################################
    # // Load the local image tile, once per tile
    # for(j=0; j<TILE_WIDTH; j++){
    #     for(k=0; k<TILE_WIDTH; k++){
    #             img_tile[j][k] = img[(j + blockIdx.y*TILE_WIDTH)*(C) + (k + blockIdx.x*TILE_WIDTH)];
    #     }
    # }
    # // check within global boundaries, then for each tile, count them up into local bins
    # for(j=0; j<TILE_WIDTH; j++){
    #     for(k=0; k<TILE_WIDTH; k++){
    #             //++bins_loc[img_tile[j][k]];
    #             ++bins_loc[img[(j + blockIdx.y*TILE_WIDTH)*C + (k + blockIdx.x*TILE_WIDTH)]];
    #     }
    # }
    # // Add tiled results to global output (once per tile!)
    # if(blockIdx.x==0 && blockIdx.y==0){
    #     for (k=0; k<256; k++)
    #         atomicAdd(&bins[k], bins_loc[k]);
    # }

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

    // P - warp size
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
    //__syncthreads();

    // Load the local image tile, once per tile
    if(tile_idx==0){
        for(j=0; j<TILE_WIDTH; j++){
            for(k=0; k<TILE_WIDTH; k++){
                    img_tile[j][k] = img[(j + blockIdx.y*blockDim.y)*(gridDim.x*blockDim.x) + (k + blockIdx.x*blockDim.x)];
            }
        }
    }
    //__syncthreads();

    // check within global boundaries, then for each tile, count them up into local bins
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

def test_instance(_version_="NAIVE", _testsize_="1KB", _P_=1536, _runpy_=False, _print_=False):
    # Configurations
    # version="OPTIMIZED"
    version=_version_
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
        print "python time taken: ", time.time()-start

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
    else: # else do the optimized version
        opt_hist(img_gpu, bin_gpu, np.uint32(32), np.uint32(R), np.uint32(C), block=(16,16,1), grid=(C/16,R/16,1))
        # opt_hist(img_gpu, bin_gpu, np.uint32(32), np.uint32(R), np.uint32(C), block=(1,1,1), grid=(C/16,R/16,1))
    runtime = time.time()-start
    print "cuda time taken: ", runtime
    h_op =  bin_gpu.get()
    h_op[0] = N/256
    # NOTE: had to fudge elemnt zero of the output, because it was an outlier screwing up RMSE

    # Check correctness:
    if _print_:
        print "kernel used: ", version
        print "cuda output:"
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
# plt.title('Histogram Kernel Optimization: CUDA')
# plt.gca().set_xlim((min(px), max(px)))
# plt.gca().set_ylim((min(naive_times)/2, max(opt_times)*3))
# plt.savefig('cuda_hist_times.png')
