#########################################################################################################################
# HEADER
#
# Filename:    cipher.py
# Description: Implementation of Assignment 1
#
# Owner:       Alexander Stein
# Date:        9/28/2016
# School:      Columbia University
# Class:       EECSE4750
#########################################################################################################################

#########################################################################################################################
# References
#
# Intro to PyOpenCL, Gaston Hillar - http://www.drdobbs.com/open-source/easy-opencl-with-python/240162614?pgno=2
# Intro to OpenCL, Matthew Scarpino - http://www.drdobbs.com/parallel/a-gentle-introduction-to-opencl/231002854?pgno=3
# PyCUDA Examples, Andreas Klockner - https://wiki.tiker.net/PyCuda/Examples/ArithmeticExample
#
#########################################################################################################################

#########################################################################################################################
# IMPORT STATEMENTS
#########################################################################################################################

# Imports for PyOpenCL
import pyopencl as cl
from pyopencl import array

# Imports for PyCUDA
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule

# General Imports
import numpy as np
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

class CaseException(Exception):
	pass

#########################################################################################################################
# Function Definitions
#########################################################################################################################

def atbashEncodePY(input_string):

	'''applies cipher to an input string'''

	# map the input to output using the cipher
	output = ""

	start = time.time()
	for idx, val in enumerate(input_string):
		ascii = ord(val)
		output += chr(ascii + 25 - 2 * (ascii - 65))
	runtime = time.time() - start

	return runtime, output


def atbashEncodeOCL(input_string):
	# 1. Obtain OpenCL Platform
	platform = cl.get_platforms()[0]

	# 2. Obtain Device ID for GPU
	device_id = platform.get_devices()[0]

	# 3. Create Context for selected device
	context = cl.Context([device_id])

	# 4. Create a program for the context, give it a kernel, and build
	program = cl.Program(context, """
		__kernel void apply_cipher(__global const char *inputString, __global int *len, __global char *outputString)
		{
			// This input will be a 1-D array, so we only need a single ID to work with it (function of block grid)

				int gid_x = get_global_id(0);
				int gdim_x = get_global_size(0);
				int gid_y = get_global_id(1);

			// ATBASH Encoding can be easily accomplished with some simple ASCII Algebra
			// A = 65, Z = 90, A-Z = 25, e.g. B-Y = 23, C-X = 21 ... ASCII(OUT) = ASCII(IN) + 25 - (2 * (ASCII(IN) - 65))

				int ASCII_VALUE = 0;

				// total avaialbe threads may be more than needed

				if ((gdim_x * gid_y + gid_x) < len){
					ASCII_VALUE = (int) inputString[gdim_x * gid_y + gid_x];
					outputString[gdim_x * gid_y + gid_x] = ASCII_VALUE + 25 - 2 * (ASCII_VALUE - 65);
				}// end if

		}// end kernel
		""").build()

	# 5. Create a command queue for the target device
	queue = cl.CommandQueue(context)

	# 6. Allocate device memory and move input data from the host to the device memory.
	mem_flags = cl.mem_flags
	length = len(input_string)
	size = np.float32(length)

	# max_size = device_id.max_work_group_size
	max_size = 1024

	len_buf = cl.Buffer(context, mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR, hostbuf=size)
	inputString_buf = cl.Buffer(context, mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR, hostbuf=input_string)
	output_string = np.empty_like(input_string)
	outputString_buf = cl.Buffer(context, mem_flags.WRITE_ONLY, sys.getsizeof(input_string))

	# 7. Map buffers to kernel arguments and deploy the kernel, with specified local and global dimensions
	global_dim = (max_size, int(math.ceil(length/float(max_size))))
	local_dim = (max_size,1)

	### Time the function call
	start = time.time()
	program.apply_cipher(queue, global_dim, local_dim, inputString_buf, len_buf, outputString_buf)
	runtime = time.time() - start

	# 8. Move the kernel's output data back to the host memory
	cl.enqueue_copy(queue, output_string, outputString_buf)

	return runtime, output_string


def atbashEncodeCUDA(input_string):
	op = np.empty_like(input_string)
	n = len(input_string)

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

	# max_size = pycuda.autoinit.device.get_attribute(max_threads_per_block)
	max_size = 1024

	size = n * sys.getsizeof('A')

	inp_gpu = cuda.mem_alloc(size)
	op_gpu = cuda.mem_alloc(size)

	cuda.memcpy_htod(inp_gpu, input_string)
	cuda.memcpy_htod(op_gpu, op)

	b_size = n if (n <= max_size) else max_size

	start = time.time()
	apply_cipher(inp_gpu, op_gpu, np.uint32(n), block=(b_size, 1, 1))
	runtime = time.time() - start

	cuda.memcpy_dtoh(op, op_gpu)


	return runtime, op

def nTest(input_string, n, lang="PYTHON"):
	# n_string = ''.join(random.choice(string.ascii_uppercase) for _ in range(n))
	# print "nString: ", n_string

	n_string = input_string

	if ( lang == "PYTHON" ):
		runtime, output = atbashEncodePY(n_string)
	elif ( lang == "OPENCL" ):
		runtime, output = atbashEncodeOCL(n_string)
	elif ( lang == "CUDA" ):
		runtime, output = atbashEncodeCUDA(n_string)

	return runtime, output

#########################################################################################################################
# Main
#########################################################################################################################


def main(_repetitions_=1):

	'''Default arguments: _repetitions_ - how many times to repeat input string'''

	# input_string = "ALEX"

	# Generate a random uppercase ASCII string of length _repetitions_
	input_string = ''.join(random.choice(string.ascii_uppercase) for _ in range(_repetitions_))
	# print "INPUT: ", input_string

	pytime, pyout = nTest(input_string, _repetitions_, "PYTHON")
	ocltime, oclout = nTest(input_string, _repetitions_, "OPENCL")
	ctime, cout = nTest(input_string, _repetitions_, "CUDA")

	print "PYTHON results: ", pytime, pyout
	print "OpenCL results: ", ocltime, oclout
	print "CUDA results: ", ctime, cout

	print "Outputs Equal." if (pyout == str(oclout)) and (pyout == str(cout)) else "Outputs Unequal."

	TEST_ALL = False
	if TEST_ALL :
		M = 3
		python_times = []
		ocl_times = []
		cuda_times = []

		for k in xrange(1,_repetitions_) :

			input_string = ''.join(random.choice(string.ascii_uppercase) for _ in range(k))

			python_times_tmp = []
			ocl_times_tmp = []
			cuda_times_tmp = []

			for i in xrange(M):

				pytime, pyout = nTest(input_string, k, "PYTHON")
				ocltime, oclout = nTest(input_string, k, "OPENCL")
				ctime, cout = nTest(input_string, k, "CUDA")

				python_times_tmp.append(pytime)
				ocl_times_tmp.append(ocltime)
				cuda_times_tmp.append(ctime)

			python_times.append(np.average(python_times_tmp))
			ocl_times.append(np.average(ocl_times_tmp))
			cuda_times.append(np.average(cuda_times_tmp))

		MAKE_PLOT = True
		if MAKE_PLOT:
			import matplotlib as mpl
			mpl.use('agg')
			import matplotlib.pyplot as plt
			x = list(xrange(1,_repetitions_))

			plt.gcf()
			plt.plot(x, python_times, color='r', label='python')
			plt.plot(x, ocl_times, color='b', label='openCL')
			plt.plot(x, cuda_times, color='g', label='CUDA')
			plt.xlabel('length')
			plt.ylabel('time')
			plt.legend(loc='upper left')
			plt.gca().set_xlim((min(x), max(x)))
			plt.gca().set_ylim((min(python_times)/2, 2*max(ocl_times)))
			plt.savefig('python_times.png')


if __name__ == '__main__':
	main(int(sys.argv[1]))
