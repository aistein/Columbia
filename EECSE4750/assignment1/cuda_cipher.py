#########################################################################################################################
# HEADER
#
# Filename:    cuda_cipher.py
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
# desired number.  To produce a graph, set TEST_ALL=True.
#
#########################################################################################################################
#########################################################################################################################
# References
#
# PyCUDA Examples, Andreas Klockner - https://wiki.tiker.net/PyCuda/Examples/ArithmeticExample
#
#########################################################################################################################

#########################################################################################################################
# IMPORT STATEMENTS
#########################################################################################################################

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

def atbashEncodeCUDA(input_string):

### 1,2.3. Platform Info, Device Info, and Context are all obtained with the import statement "import pycuda.autoinit"

### 4. Create a program for the context, give it a kernel, and build
    mod = SourceModule("""
        __global__ void apply_cipher(const char* inp, char* op, int len)
        {
            // This input will be a 2-D matrix, so we only need three pieces of info to work with it (function of block grid)
    	        int tidx = threadIdx.x;
    			int bidx = blockIdx.x;
    			int bdimx = blockDim.x;

    		// ATBASH Encoding can be easily accomplished with some simple ASCII Algebra
    		// A = 65, Z = 90, A-Z = 25, e.g. B-Y = 23, C-X = 21 ... ASCII(OUT) = ASCII(IN) + 25 - (2 * (ASCII(IN) - 65))

    		// total avaialbe threads may be more than needed.
            // They can be trimmed with the knowledge that matrices are "flattened," i.e. actually stored as a 1-D
            // contiguous memory array.  Thus, to access the right element of the matrix in the format of the 1-D
            // array, we need to calculate the index "oid_x = bdim_x * bid_x + tid_x"

    			int oidx = bidx * bdimx + tidx;
    	        int ASCII = 0;

    	        if ( oidx <= len ){
    	            ASCII = inp[oidx];
    	            op[oidx] = ASCII + 25 - 2 * ( ASCII - 65 );
    	        }
        }
        """)
    apply_cipher = mod.get_function("apply_cipher")

    ### 5. Command Queue is also handled with import statement "import pycuda.autoinit"

    ### 6. Allocate device memory and move input data from the host to the device memory.
    op = np.empty_like(input_string)
    n = len(input_string)
    max_size = 1024
    size = n * sys.getsizeof('A')

    inp_gpu = cuda.mem_alloc(size)
    op_gpu = cuda.mem_alloc(size)

    cuda.memcpy_htod(inp_gpu, input_string)
    cuda.memcpy_htod(op_gpu, op)

    ### 7. Map buffers to kernel arguments and deploy the kernel, with specified block and grid dimensions
    ###        CUDA organizes memory into a "grid of blocks containing threads."
    ###        Here the grid is 1-D, as are the blocks, each containing 1024 threads.

    b_size = n if (n <= max_size) else max_size
    g_size = 1 if (n <= max_size) else int(math.ceil(n/float(max_size)))

    ## Time the deployment of the kernel for metrics
    start = time.time()
    apply_cipher(inp_gpu, op_gpu, np.uint32(n), block=(b_size, 1, 1), grid=(g_size, 1, 1))
    runtime = time.time() - start

    ### 8. Move the kernel's output data back to the host memory
    cuda.memcpy_dtoh(op, op_gpu)


    return runtime, op

def nTest(input_string, n, lang="PYTHON"):

    ''' This function is just a buffer here, but can be modified to run all types of encoding in a single file '''

    n_string = input_string

    if ( lang == "PYTHON" ):
    	runtime, output = atbashEncodePY(n_string)
    # elif ( lang == "OPENCL" ):
    # 	runtime, output = atbashEncodeOCL(n_string)
    elif ( lang == "CUDA" ):
    	runtime, output = atbashEncodeCUDA(n_string)

    return runtime, output

#########################################################################################################################
# Main
#########################################################################################################################

def main(_repetitions_=1):

    '''Default arguments: _repetitions_ - how many times to repeat input string'''

    TEST_ALL = False

    ### not TEST_ALL :
    ###     Don't collect any data, just print the results for a single run on each method for a string of
    ###     length _repetitions_.  Test to ensure that the outputs are equal
    if (not TEST_ALL):

		### Part 1:  Input your name as an array of characters. Implement this simple cipher on Python.
		name = "ALEX"
		print "PART 1 (PYTHON): ", name," --> ", str(atbashEncodePY(name)[1])

        ### Part 2: Implement the same using CUDA.
		print "PART 2 (CUDA): ", name," --> ", str(atbashEncodeCUDA(name)[1])

		### Part 3: Repeat 1 & 2 With Random String Lengths and Contents

        ### Generate a random uppercase ASCII string of length _repetitions_
		input_string = ''.join(random.choice(string.ascii_uppercase) for _ in range(_repetitions_))

        ### Run the tests
		pytime, pyout = nTest(input_string, _repetitions_, "PYTHON")
		ctime, cout = nTest(input_string, _repetitions_, "CUDA")

		print "PART 3a (PYTHON): ", input_string," --> ", pyout
		print "PYTHON results: ", pytime
		print "PART 3b (CUDA): ", input_string," --> ", cout
		print "CUDA results: ", ctime

        ### Test to be sure the outputs of each method of encoding produce the same results
		print "Outputs Equal." if (pyout == str(cout)) else "Outputs Unequal."

    ### TEST_ALL :
    ###     Collect data on running time it takes to run the cipher in both methods from string length 1
    ###     all the way up to _repetitions_.  Produce a graph if desired
    if TEST_ALL :
        M = 3
        python_times = []
        cuda_times = []

        for k in xrange(1,_repetitions_) :

            ### Generate a random uppercase ASCII string of length _repetitions_
        	input_string = ''.join(random.choice(string.ascii_uppercase) for _ in range(k))

        	python_times_tmp = []
        	cuda_times_tmp = []

            ### Average over M results on the same string to reduce outliers
        	for i in xrange(M):

                ### Run the tests, store the results
        		pytime, pyout = nTest(input_string, k, "PYTHON")
        		ctime, cout = nTest(input_string, k, "CUDA")

        		python_times_tmp.append(pytime)
        		cuda_times_tmp.append(ctime)

        	python_times.append(np.average(python_times_tmp))
        	cuda_times.append(np.average(cuda_times_tmp))

        MAKE_PLOT = True
        if MAKE_PLOT:
        	import matplotlib as mpl
        	mpl.use('agg')
        	import matplotlib.pyplot as plt
        	px = list(xrange(len(python_times)))
        	cx = list(xrange(len(cuda_times)))

        	plt.gcf()
        	plt.plot(px, python_times, color='r', label='python')
        	plt.plot(cx, cuda_times, color='g', label='CUDA')
        	plt.xlabel('length of string')
        	plt.ylabel('time')
        	plt.legend(loc='upper left')
        	plt.gca().set_xlim((min(px), max(px)))
        	plt.gca().set_ylim((min(python_times)/2, max(python_times)*1.2))
        	plt.savefig('python_v_cuda_times.png')


if __name__ == '__main__':
	main(20)
