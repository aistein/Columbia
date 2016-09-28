import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule

import numpy
import sys

inp = "ALEXISTHEBESTEVER"
op = numpy.empty_like(inp)
n = len(inp)
size = n * sys.getsizeof(' ')

inp_gpu = cuda.mem_alloc(n * size)
op_gpu = cuda.mem_alloc(n * size)
#n_gpu = cuda.mem_alloc(sys.getsizeof(0))

cuda.memcpy_htod(inp_gpu, inp)
cuda.memcpy_htod(op_gpu, op)
#cuda.memcpy_htod(n_gpu, n)

mod = SourceModule("""
    __global__ void apply_cipher(const char* inp, char* op, int len)
    {
        int idx = threadIdx.x;
        float ASCII = 0.0;
        if ( idx <= len ){
            ASCII = (float) inp[idx];
            op[idx] = ASCII + 25.0 - 2 * ( ASCII - 65.0 );
        }
    }
    """)

apply_cipher = mod.get_function("apply_cipher")
apply_cipher(inp_gpu, op_gpu, numpy.uint32(n), block=(n,1,1))

cuda.memcpy_dtoh(op, op_gpu)

print op


