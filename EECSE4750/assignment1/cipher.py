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
# IMPORT STATEMENTS
#########################################################################################################################

import string
import pyopencl as cl

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

def atbashInit(plaintext = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"):
	
	'''initiates the atbash cipher mapping into returned dictionary'''

	# Reverse the alphabet into a new array (can be done with any alphabet)
	ciphertext = []
	for idx, val in enumerate(plaintext):
		ciphertext.insert(0,val)

	# Map the arrays together with a dictionary
	cipherdict={}
	for idx, val in enumerate(plaintext):
		cipherdict[val] = ciphertext[idx]

	return cipherdict


def atbashEncode(inputStr):

	'''applies cipher to an input string'''

	# initialize the cipher
	cipher = atbashInit()

	# map the input to output using the cipher
	output = ""

	for letter in inputStr:
		try:
			if not str.isupper(cipher[letter]):
				raise CaseException, inputStr
			else:
				output += cipher[letter]
		except CaseException, inputStr:
			print "-E- Input contains lower-case letters: %s\n", inputStr

	return output


#########################################################################################################################
# Main
#########################################################################################################################


def main(_outdir_="/etl_output/"):

	'''Default arguments: _outdir_ - where output files will go'''

	name = "ALEX"
	jumble = atbashEncode(name)

	platforms = cl.get_platforms()

	print jumble
	print platforms[0].get_devices()[0].global_mem_size

if __name__ == '__main__':
	main()