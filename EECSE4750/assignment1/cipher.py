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
import math

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

	for idx, val in enumerate(input_string):
		ascii = ord(val)
		output += chr(ascii + 25 - 2 * (ascii - 65))

	print "PY output: ", output

	return


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
			// This input will be a 1-D array, so we only need a single ID to work with it

				int gid_x = get_global_id(0);
				int gdim_x = get_global_size(0);
				int gid_y = get_global_id(1);

			// ATBASH Encoding can be easily accomplished with some simple ASCII Algebra
			// A = 65, Z = 90, A-Z = 25, B-Y = 23, C-X = 21 ... ASCII(OUT) = ASCII(IN) + 25 - (2 * (ASCII(IN) - 65))

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
	size = np.float32(len(input_string))
	max_size = device_id.max_work_group_size
	if (size <= max_size):
		len_buf = cl.Buffer(context, mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR, hostbuf=size)
		inputString_buf = cl.Buffer(context, mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR, hostbuf=input_string)
		output_string = np.empty_like(input_string)
		outputString_buf = cl.Buffer(context, mem_flags.WRITE_ONLY, sys.getsizeof(output_string))

		# 7. Map buffers to kernel arguments and deploy the kernel, with specified local and global dimensions
		global_dim = (len(input_string),) if (len(input_string) <= 16) else (16, int(math.ceil(len(input_string)/16.0)))
		local_dim = None if (len(input_string) < 16) else (4,int(math.ceil(len(input_string)/16.0))/2)
		program.apply_cipher(queue, global_dim, local_dim, inputString_buf, len_buf, outputString_buf)

		# 8. Move the kernel's output data back to the host memory
		cl.enqueue_copy(queue, output_string, outputString_buf)

		print "OpenCL output: ", output_string

	# need to account for possibility of exceeding maximum work_group size
	elif (size > max_size):
		input_strings = [input_string[i:i+max_size] for i in range(0, len(input_string), max_size)]
		output_string_n = ""
		for input_div in input_strings:
			print len(input_div)
			len_buf = cl.Buffer(context, mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR, hostbuf=size)
			inputString_buf = cl.Buffer(context, mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR, hostbuf=input_div)
			output_string = np.empty_like(input_div)
			outputString_buf = cl.Buffer(context, mem_flags.WRITE_ONLY, sys.getsizeof(output_string))

			# 7. Map buffers to kernel arguments and deploy the kernel, with specified local and global dimensions
			global_dim = (len(input_div),) if (len(input_div) <= 16) else (16, int(math.ceil(len(input_div)/16.0)))
			local_dim = None if (len(input_div) < 16) else (4,int(math.ceil(len(input_div)/16.0))/2)
			program.apply_cipher(queue, global_dim, local_dim, inputString_buf, len_buf, outputString_buf)

			# 8. Move the kernel's output data back to the host memory
			cl.enqueue_copy(queue, output_string, outputString_buf)
			output_string_n += str(output_string)

			print "OpenCL output: ", output_string
		print "OpenCL final output N: ", output_string_n
	return


def atbashEncodeCUDA(input_string):
	op = np.empty_like(input_string)
	n = len(input_string)
	size = n * sys.getsizeof(' ')

	inp_gpu = cuda.mem_alloc(n * size)
	op_gpu = cuda.mem_alloc(n * size)

	cuda.memcpy_htod(inp_gpu, input_string)
	cuda.memcpy_htod(op_gpu, op)

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
	apply_cipher(inp_gpu, op_gpu, np.uint32(n), block=(n,1,1))

	cuda.memcpy_dtoh(op, op_gpu)
	print "CUDA output: ", op

	return

def nTest(input_string, n, lang="PYTHON"):
	n_string = ""
	for x in range(n):
		n_string += input_string

	if ( lang == "PYTHON" ):
		atbashEncodePY(n_string)
	elif ( lang == "OPENCL" ):
		atbashEncodeOCL(n_string)
	elif ( lang == "CUDA" ):
		atbashEncodeCUDA(n_string)

	return

#########################################################################################################################
# Main
#########################################################################################################################


def main(_outdir_="/etl_output/"):

	'''Default arguments: _outdir_ - where output files will go'''

	### Execute Project Task in Python
	input_string = "K"
	nTest(input_string, 2048, "OPENCL")
	# atbashEncodePY(input_string)
	# atbashEncodeOCL(input_string)
	# atbashEncodeCUDA(input_string)



if __name__ == '__main__':
	main()