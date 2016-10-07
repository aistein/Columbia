#########################################################################################################################
# HEADER
#
# Filename:    opencl_cipher.py
# Description: Implementation of Assignment 1
#
# Owner:       Alexander Stein
# Date:        10/7/2016
# School:      Columbia University
# Class:       EECSE4750
#########################################################################################################################
#########################################################################################################################
# Instructions:
#
# If you wish to increase the length of the string under test, scroll to the bottom of the file, replace "100" with
# desired number.
#
#########################################################################################################################
#########################################################################################################################
# References
#
# Intro to PyOpenCL, Gaston Hillar - http://www.drdobbs.com/open-source/easy-opencl-with-python/240162614?pgno=2
# Intro to OpenCL, Matthew Scarpino - http://www.drdobbs.com/parallel/a-gentle-introduction-to-opencl/231002854?pgno=3
#
#########################################################################################################################

#########################################################################################################################
# IMPORT STATEMENTS
#########################################################################################################################

# Imports for PyOpenCL
import pyopencl as cl
from pyopencl import array

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
	### 1. Obtain OpenCL Platform
	platform = cl.get_platforms()[0]

	### 2. Obtain Device ID for GPU
	device_id = platform.get_devices()[0]

	### 3. Create Context for selected device
	context = cl.Context([device_id])

	### 4. Create a program for the context, give it a kernel, and build
	program = cl.Program(context, """
		__kernel void apply_cipher(__global const char *inputString, __global int *len, __global char *outputString)
		{
			// This input will be a 2-D matrix, so we only need three pieces of info to work with it (function of block grid)

				int gid_x = get_global_id(0); // global ROW ID
				int gdim_x = get_global_size(0); // global ROW dimensions
				int gid_y = get_global_id(1); // global COLUMN ID

			// ATBASH Encoding can be easily accomplished with some simple ASCII Algebra
			// A = 65, Z = 90, A-Z = 25, e.g. B-Y = 23, C-X = 21 ... ASCII(OUT) = ASCII(IN) + 25 - (2 * (ASCII(IN) - 65))

			// total avaialbe threads may be more than needed.
            // They can be trimmed with the knowledge that matrices are "flattened," i.e. actually stored as a 1-D
            // contiguous memory array.  Thus, to access the right element of the matrix in the format of the 1-D
            // array, we need to calculate the index "oid_x = gdim_x * gid_y + gid_x"

                int oid_x = gdim_x * gid_y + gid_x;
				int ASCII = 0;

				if ((oid_x) < len){
					ASCII = (int) inputString[oid_x];
					outputString[oid_x] = ASCII + 25 - 2 * (ASCII - 65);
				}// end if

		}// end kernel
		""").build()

	### 5. Create a command queue for the target device
	queue = cl.CommandQueue(context)

	### 6. Allocate device memory and move input data from the host to the device memory.
	mem_flags = cl.mem_flags
	length = len(input_string)
	size = np.float32(length)

	max_size = 1024

	len_buf = cl.Buffer(context, mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR, hostbuf=size)
	inputString_buf = cl.Buffer(context, mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR, hostbuf=input_string)
	output_string = np.empty_like(input_string)
	outputString_buf = cl.Buffer(context, mem_flags.WRITE_ONLY, sys.getsizeof(input_string))

	### 7. Map buffers to kernel arguments and deploy the kernel, with specified local and global dimensions
    ###    Choosing Dimensions:
    ###        A given work-group cannot exceed 1024 work-items; if > 1024 are required, A new rank is created
    ###        Excess work-item space is trimmed in the kernel with 'if ((gdim_x * gid_y + gid_x) < len){''

	global_dim = (max_size, int(math.ceil(length/float(max_size))))
	local_dim = (max_size,1)

	## Time the deployment of the kernel for metrics
	start = time.time()
	program.apply_cipher(queue, global_dim, local_dim, inputString_buf, len_buf, outputString_buf)
	runtime = time.time() - start

	### 8. Move the kernel's output data back to the host memory
	cl.enqueue_copy(queue, output_string, outputString_buf)

	return runtime, output_string

def nTest(input_string, n, lang="PYTHON"):

    ''' This function is just a buffer here, but can be modified to run all types of encoding in a single file '''

	n_string = input_string

	if ( lang == "PYTHON" ):
		runtime, output = atbashEncodePY(n_string)
	elif ( lang == "OPENCL" ):
		runtime, output = atbashEncodeOCL(n_string)
	# elif ( lang == "CUDA" ):
	# 	runtime, output = atbashEncodeCUDA(n_string)

	return runtime, output

#########################################################################################################################
# Main
#########################################################################################################################

def main(_repetitions_=1):

	'''Default arguments: _repetitions_ - how many times to repeat input string'''

	TEST_ALL = True

    ### not TEST_ALL :
    ###     Don't collect any data, just print the results for a single run on each method for a string of
    ###     length _repetitions_.  Test to ensure that the outputs are equal
    if (not TEST_ALL):
    	# Generate a random uppercase ASCII string of length _repetitions_
    	input_string = ''.join(random.choice(string.ascii_uppercase) for _ in range(_repetitions_))

        ### Run the tests
    	pytime, pyout = nTest(input_string, _repetitions_, "PYTHON")
    	ocltime, oclout = nTest(input_string, _repetitions_, "OPENCL")

    	print "PYTHON results: ", pytime
    	print "OpenCL results: ", ocltime

        ### Test to be sure the outputs of each method of encoding produce the same results
    	print "Outputs Equal." if (pyout == str(oclout)) else "Outputs Unequal."

    ### TEST_ALL :
    ###     Collect data on running time it takes to run the cipher in both methods from string length 1
    ###     all the way up to _repetitions_.  Produce a graph if desired
	if TEST_ALL :
		M = 3
		python_times = []
		ocl_times = []

		for k in xrange(1,_repetitions_) :

            ### Generate a random uppercase ASCII string of length _repetitions_
			input_string = ''.join(random.choice(string.ascii_uppercase) for _ in range(k))

			python_times_tmp = []
			ocl_times_tmp = []

            ### Average over M results on the same string to reduce outliers
			for i in xrange(M):

                ### Run the tests, store the results
				pytime, pyout = nTest(input_string, k, "PYTHON")
				ocltime, oclout = nTest(input_string, k, "OPENCL")

				python_times_tmp.append(pytime)
				ocl_times_tmp.append(ocltime)

			python_times.append(np.average(python_times_tmp))
			ocl_times.append(np.average(ocl_times_tmp))

		### Normalize the Data for good graphs
		python_times_norm = [float(val)/max(python_times) for val in python_times]
		ocl_times_norm = [float(val)/max(ocl_times) for val in ocl_times]

		MAKE_PLOT = True
		if MAKE_PLOT:
			import matplotlib as mpl
			mpl.use('agg')
			import matplotlib.pyplot as plt
			px = list(xrange(len(python_times_norm)))
			ox = list(xrange(len(ocl_times_norm)))

			plt.gcf()
			plt.plot(px, python_times_norm, color='r', label='python')
			plt.plot(ox, ocl_times_norm, color='b', label='openCL')
			plt.xlabel('length of string')
			plt.ylabel('normalized time')
			plt.legend(loc='upper left')
			plt.gca().set_xlim((min(px), max(px)))
			plt.gca().set_ylim((min(python_times_norm)/2, 2*max(ocl_times_norm)))
			plt.savefig('python_v_ocl_times.png')


if __name__ == '__main__':
	main(100)
