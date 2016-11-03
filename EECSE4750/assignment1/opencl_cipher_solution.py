


"""
 Cipher encoding in PyOpenCL
"""

#the following module is used to mark the time stamps
import time

#import OpenCL library in the python environment
import pyopencl as cl
import pyopencl.array
#for use of arrays
import numpy as np
#useful for system related functions like read/write etc
import sys

# Select the desired OpenCL platform; you shouldn't need to change this:
NAME = 'NVIDIA CUDA'
platforms = cl.get_platforms()
devs = None
for platform in platforms:
    if platform.name == NAME:
        devs = platform.get_devices()

# Set up a command queue:
ctx = cl.Context(devs)
queue = cl.CommandQueue(ctx, properties=cl.command_queue_properties.PROFILING_ENABLE)

#GPU environment - OpenCL starts here
kernel = """
__kernel void cipher(__global char* c, __global char *d){

  int i = get_global_id(0);  //this 'i' instead of becoming iterator in for loop now acts as threadID for same process from i=0 to size(c)
  d[i] = 90 - c[i] + 65;  //take advantage of ASCII table and type conversion

}

"""


###################################################################################
##Python code starts here
############################################################################

#run the scheme into loop

x = np.arange(0,400,1) #this code makes 100 iterations

#define timer arrays - one for OpenCL and one for Python times
py_times = np.zeros_like(x).astype(np.float64)
op_times = np.zeros_like(x).astype(np.float64)

#build the Kernel
prg = cl.Program(ctx, kernel).build()

#Part 3 Loop starts here (this will cover Part 1 and 2 also
for r in xrange(1,400,1):

    name = "DHRUV"*r #replicate the string r number of times
    #In Python, string is not by-default considered as numpy array. Hence, we need to convert it into np array.
    #First we need to break string into individual characters
    name = list(name)
    #Next,convert this array of characters into np array
    name = np.array(name)

    l = len(name)

    #initialize input buffer to OpenCL
    name_buf = cl.array.to_device(queue,name)

    #initialize output buffer to openCL, its dimensions should be similar to the name
    c_buf = cl.array.empty(queue, name.shape,name.dtype)

    c_gpu = np.empty(name.shape,name.dtype) #store GPU data output in python
    c_py = np.empty(name.shape,name.dtype)  #stores python output

    times=[] #There can be glitches while running the code, hence, it is sometimes better to take multiple readings for same input and take average
    for M in xrange(3):

        start = time.time()
        for i in range(0,l):
            c_py[i] = unichr(90 - ord(name[i]) + 65) #ord - converts char into int and unichr converts int into char
        times.append(time.time()-start)
    py_times[r] = np.average(times) #append the time in py_times array

    # openCL Call
    times=[]
    for M in xrange(3):
        start = time.time()
        #Call OpenCL program: kernel_func_name(queue,data SIZE, <Block dimensions>, input data, output data)
        prg.cipher(queue, name.shape, None, name_buf.data, c_buf.data)
        times.append(time.time()-start)
    op_times[r] = np.average(times)

    c_gpu = c_buf.get() #get the data from c_buf
    print 'c_py=',c_py,' and c_gpu = ',c_gpu

    print 'coded equality:        ', (c_py==c_gpu).all()
    print 'string_len=',r*5,'   py time:',py_times[r],' op time:',op_times[r]

# Optional: if you want to plot the function, set MAKE_PLOT to
# True:
MAKE_PLOT = True
if MAKE_PLOT:
    import matplotlib as mpl
    mpl.use('agg')
    import matplotlib.pyplot as plt

    plt.gcf()
    #plt.subplot(211)
    plt.plot(5*x, py_times,'r') #5 is length of original name
    plt.plot(5*x, op_times,'g')
    plt.legend(['python cipher', 'opencl cipher'], loc='upper left')
    plt.xlabel('length of string')
    plt.ylabel('output coding times')
    plt.gca().set_xlim((min(5*x), max(5*x)))


   #plt.gcf()
    plt.savefig('cipher_plots.png')
