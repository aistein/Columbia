/* Vector addition using OpenCL. */
 
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <CL/cl.h>
 
const char *kernel_src =
"    __kernel void func(__global float* a, __global float* b, __global float* c) {\n"
"    unsigned int i = get_global_id(0);\n"
"    c[i] = a[i]+b[i];\n"
"}\n";
 
void vector_add(float x[], float y[], float z[], unsigned int N) {
    unsigned int i;
    for (i=0; i<N; i++)
        z[i] = x[i]+y[i];
}
 
int main(void) {
 
    /* Get available platforms: */
    cl_platform_id *platforms = NULL;
    cl_uint num_platforms;
 
    cl_int clStatus = clGetPlatformIDs(0, NULL, &num_platforms);
    platforms = (cl_platform_id *) malloc(sizeof(cl_platform_id)*num_platforms);
    clStatus = clGetPlatformIDs(num_platforms, platforms, NULL);
 
    /* Find the NVIDIA CUDA platform: */
    unsigned int i;
    char queryBuffer[1024];
    for (i=0; i<num_platforms; i++) {
        clStatus = clGetPlatformInfo(platforms[i], CL_PLATFORM_NAME,
                          1024, &queryBuffer, NULL);
        if (strcmp(queryBuffer, "NVIDIA CUDA") == 0)
            break;
    }
 
    /* Get the GPU devices for the selected platform: */
    cl_device_id *device_list = NULL;
    cl_uint num_devices;
    clStatus = clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_GPU,
                              0, NULL, &num_devices);
    device_list = (cl_device_id *) malloc(sizeof(cl_device_id)*num_devices);
    clStatus = clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_GPU,
                              num_devices, device_list, NULL);
 
    /* Create context: */
    cl_context context;
    context = clCreateContext(NULL, num_devices, device_list,
                              NULL, NULL, &clStatus);
 
    /* Create command queue: */
    cl_command_queue command_queue = clCreateCommandQueue(context, device_list[0],
                                                          0, &clStatus);
 
    /* Load some random data to process: */
    unsigned int N = 100000;
    float a[N];
    float b[N];
    for (i=0; i<N; i++) {
        a[i] = (float) rand()/(float) RAND_MAX;
        b[i] = (float) rand()/(float) RAND_MAX;
    }
 
    /* Create buffers: */
    cl_mem a_buf = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                  N*sizeof(float), a, &clStatus);
    cl_mem b_buf = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                  N*sizeof(float), b, &clStatus);
    cl_mem c_buf = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
                                  N*sizeof(float), NULL, &clStatus);
 
    /* Create program: */
    cl_program program = clCreateProgramWithSource(context, 1, (const char **) &kernel_src,
                                                   NULL, &clStatus);
 
    /* Build kernel: */
    cl_kernel kernel = clCreateKernel(program, "kernel", &clStatus);
 
    /* Set the kernel arguments: */
    clStatus = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *) &a_buf);
    clStatus = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *) &b_buf);
    clStatus = clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *) &c_buf);
 
    /* Execute the kernel: */
    size_t global_size = N;
    size_t local_size = 1;
    clStatus = clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL,
                                      &global_size, &local_size, 0, NULL, NULL);
 
    /* Retrieve the result: */
    float c[N];
    clStatus = clEnqueueReadBuffer(command_queue, c_buf, CL_TRUE, 0,
                                   N*sizeof(float), c, 0, NULL, NULL);
 
    /* Wait for all commands in the queue to complete: */
    clStatus = clFlush(command_queue);
    clStatus = clFinish(command_queue);
 
    /* Compute result using Python: */
    float c_py[N];
    vector_add(a, b, c_py, N);
 
    /* Verify that the result is correct: */
    unsigned int is_true = 0;
    for (i=0; i<N; i++) {
        if (c[i] != c_py[i]) {
            is_true = 1;
            break;
        }
    }
    printf("equal:      ");
    if (is_true) {
        printf("True\n");
    } else {
        printf("False\n");
    }
 
    /* Compare performance: */
    unsigned int M = 3;
    clock_t timing = 0;
    clock_t start;
    for (i=0; i<M; i++) {
        start = clock();
        vector_add(a, b, c_py, N);
        timing += clock()-start;
    }
    printf("c time:      %.10f\n", (double)timing/(CLOCKS_PER_SEC*M));
 
    timing = 0;
    for (i=0; i<M; i++) {
        start = clock();
        clStatus = clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL,
                                          &global_size, &local_size, 0, NULL, NULL);
        clStatus = clFlush(command_queue);
        clStatus = clFinish(command_queue);
        timing += clock()-start;
    }
    printf("opencl time: %.10f\n", (double)timing/(CLOCKS_PER_SEC*M));
 
    /* Clean up: */
    clStatus = clReleaseKernel(kernel);
    clStatus = clReleaseProgram(program);
    clStatus = clReleaseMemObject(c_buf);
    clStatus = clReleaseMemObject(b_buf);
    clStatus = clReleaseMemObject(a_buf);
    clStatus = clReleaseCommandQueue(command_queue);
    clStatus = clReleaseContext(context);
    free(device_list);
    free(platforms);
    return 0;
}
