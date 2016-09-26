#########################################################################################################################
# HEADER
#
# Filename:    cipher.py
# Description: Implementation of Assignment 1
# 
# Owner:       Alexander Stein
# Date:        9/24/2016
# School:      Columbia University
# Class:       EECSE4750
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

import string
import pyopencl as cl
from pyopencl import array
import numpy

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

def atbashInit(plaintext = "ABCDEFGHIJKLMNOPQRSTUVWXYZ", size=26):
	
	'''initiates the atbash cipher mapping into returned 2x26 Matrix'''

	# Reverse the alphabet into a new array (can be done with any alphabet)
	ciphertext = ""
	for letter in plaintext:
		ciphertext = letter + ciphertext

	# Map the arrays together
	ciphermap = plaintext + ciphertext

	print ciphermap
	return ciphermap


def atbashEncode(inputStr, cipher):

	'''applies cipher to an input string'''

	# map the input to output using the cipher
	output = ""

	for letter in inputStr:
		try:
			if not str.isupper(letter):
				raise CaseException, inputStr
			else:
				idx = cipher.index(letter)
				print idx
				output += cipher[len(cipher)/2 + idx]
		except CaseException, inputStr:
			print "-E- Input contains lower-case letters: %s\n", inputStr

	return output

#########################################################################################################################
# Main
#########################################################################################################################


def main(_outdir_="/etl_output/"):

	'''Default arguments: _outdir_ - where output files will go'''

	### Execute Project Task in Python
	cipher_map = atbashInit()
	jumble = atbashEncode("ALEX", cipher_map)
	print jumble

	### Execute Project Task in PyOpenCL (12 Steps)
	vector = numpy.zeros((1, 1), cl.array.vec.float4)
	matrix = numpy.zeros((1, 4), cl.array.vec.float4)
	matrix[0, 0] = (1, 2, 4, 8)
	matrix[0, 1] = (16, 32, 64, 128)
	matrix[0, 2] = (3, 6, 9, 12)
	matrix[0, 3] = (5, 10, 15, 25)
	vector[0, 0] = (1, 2, 4, 8)

    # 1. Obtain an OpenCL platform.
	platform = cl.get_platforms()[0]

    # 2. Obtain a device id for at least one device (accelerator).
	device_id = platform.get_devices()[0]

    # 3. Create a context for the selected device or devices.
	context = cl.Context([device_id])

    # 4. Create the accelerator program from source code.
    # 5. Build the program.

    # 6. Create one or more kernels from the program functions.
	# program = cl.Program(context, """
	# 	__kernel void matrix_dot_vector(__global const float4 *matrix,
	# 	__global const float4 *vector, __global float *result)
	# 	{
	# 		int gid = get_global_id(0);
	# 		result[gid] = dot(matrix[gid], vector[0]);
	# 	}
	# 	""").build()
	program = cl.Program(context, """
		__kernel void apply_cipher(__global const char *cipherMap, const int cmLen,
		__global const char *inputString, const int isLen, __global char *outputString))
		{
			int cm_id = get_local_id(0);
			int is_id = get_local_id(1);

			result = (cipherMap[cm_id] == inputString[is_id]) ? cipherMap[cm_id + cmLen];
		}
		""".build()

    # 7. Create a command queue for the target device.
	queue = cl.CommandQueue(context)

    # 8. Allocate device memory and move input data from the host to the device memory.
	mem_flags = cl.mem_flags
	matrix_buf = cl.Buffer(context, mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR, hostbuf=matrix)
	vector_buf = cl.Buffer(context, mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR, hostbuf=vector)
	matrix_dot_vector = numpy.zeros(4, numpy.float32)
	destination_buf = cl.Buffer(context, mem_flags.WRITE_ONLY, matrix_dot_vector.nbytes)

    # 9. Associate the arguments to the kernel with kernel object.
    # 10. Deploy the kernel for device execution.
	program.matrix_dot_vector(queue, matrix_dot_vector.shape, None, matrix_buf, vector_buf, destination_buf)

    # 11. Move the kernel's output data to host memory.
	cl.enqueue_copy(queue, matrix_dot_vector, destination_buf)

    # 12. Release context, program, kernels and memory.

	print(matrix_dot_vector)

if __name__ == '__main__':
	main()