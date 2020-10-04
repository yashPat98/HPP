//headers
#include <stdio.h>
#include <stdlib.h>                      //exit()
#include <string.h>                      //strlen()

#include <CL/opencl.h>                   //standard OpenCL header

//global OpenCL variables
cl_int            ret_ocl;               
cl_platform_id    oclPlatformID;         //compute platform id
cl_device_id      oclComputeDeviceID;    //compute device id   
cl_context        oclContext;            //compute context
cl_command_queue  oclCommandQueue;       //compute command queue
cl_program        oclProgram;            //compute program
cl_kernel         oclKernel;             //compute kernel

float *hostInput1 = NULL;
float *hostInput2 = NULL;
float *hostOutput = NULL;

//openCL memory object
cl_mem deviceInput1 = NULL;
cl_mem deviceInput2 = NULL;
cl_mem deviceOutput = NULL;

int main(void)
{
    //function declaration
    void cleanup(void);

    //variable declaration
    int inputLength;

    //code
    //hard-coded host vectors length
    inputLength = 5;

    //allocate host-memory
    hostInput1 = (float *)malloc(inputLength * sizeof(float));
    if(hostInput1 == NULL)
    {
        printf("CPU Memory Fatal Error - Can Not Allocate Enough Memory For Host Input Array 1.\nExiting ...\n");
        exit(EXIT_FAILURE);
    }

    hostInput2 = (float *)malloc(inputLength * sizeof(float));
    if(hostInput2 == NULL)
    {
        printf("CPU Memory Fatal Error - Can Not Allocate Enough Memory For Host Input Array 2.\nExiting ...\n");
        cleanup();
        exit(EXIT_FAILURE);
    }

    hostOutput = (float *)malloc(inputLength * sizeof(float));
    if(hostOutput == NULL)
    {
        printf("CPU Memory Fatal Error - Can Not Allocate Enough Memory For Host Output Array.\nExiting ...\n");
        cleanup();
        exit(EXIT_FAILURE);
    }

    //fill above input host vectors with arbitary but hard coded data 
    hostInput1[0] = 101.0f;
    hostInput1[1] = 102.0f;
    hostInput1[2] = 103.0f;
    hostInput1[3] = 104.0f;
    hostInput1[4] = 105.0f;

    hostInput2[0] = 201.0f;
    hostInput2[1] = 202.0f;
    hostInput2[2] = 203.0f;
    hostInput2[3] = 204.0f;
    hostInput2[4] = 205.0f;

    //get OpenCL supporting platform's ID
    ret_ocl = clGetPlatformIDs(1, &oclPlatformID, NULL);
    if(ret_ocl != CL_SUCCESS)
    {
        printf("OpenCL Error - clGetPlatformIDs() Failed : %d. Exiting Now ...\n", ret_ocl);
        cleanup();
        exit(EXIT_FAILURE);
    }

    //get OpenCL supporting GPU device's ID
    ret_ocl = clGetDeviceIDs(oclPlatformID, CL_DEVICE_TYPE_GPU, 1, &oclComputeDeviceID, NULL);
    if(ret_ocl != CL_SUCCESS)
    {
        printf("OpenCL Error - clGetDeviceIDs() Failed : %d. Exiting Now ...\n", ret_ocl);
        cleanup();
        exit(EXIT_FAILURE);
    }

    //create OpenCL compute context
    oclContext = clCreateContext(NULL, 1, &oclComputeDeviceID, NULL, NULL, &ret_ocl);
    if(ret_ocl != CL_SUCCESS)
    {
        printf("OpenCL Error - clCreateContext() Failed : %d. Exiting Now ...\n", ret_ocl);
        cleanup();
        exit(EXIT_FAILURE);
    }

    //create command queue
    oclCommandQueue = clCreateCommandQueue(oclContext, oclComputeDeviceID, 0, &ret_ocl);
    if(ret_ocl != CL_SUCCESS)
    {
        printf("OpenCL Error - clCreateCommandQueue() Failed : %d. Exiting Now ...\n", ret_ocl);
        cleanup();
        exit(EXIT_FAILURE);
    }

    //create OpenCL program 
    const char *oclKernelSource = 
    "__kernel void vecAdd(__global float *in1, __global float *in2, __global float *out, int len) \n" \
    "{ \n" \
        "int i = get_global_id(0); \n" \

        "if( i < len) \n" \
        "{ \n" \
            "out[i] = in1[i] + in2[i]; \n" \
        "} \n" \
    "} \n";

    oclProgram = clCreateProgramWithSource(oclContext, 1, &oclKernelSource, NULL, &ret_ocl);
    if(ret_ocl != CL_SUCCESS)
    {
        printf("OpenCL Error - clCreateProgramWithSource() Failed : %d. Exiting Now ...\n", ret_ocl);
        cleanup();
        exit(EXIT_FAILURE);
    }

    ret_ocl = clBuildProgram(oclProgram, 0, NULL, NULL, NULL, NULL);
    if(ret_ocl != CL_SUCCESS)
    {
        printf("OpenCL Error - clBuildProgram() Failed : %d. Exiting Now ...\n", ret_ocl);

        size_t len;
        char buffer[2048];
        clGetProgramBuildInfo(oclProgram, oclComputeDeviceID, CL_PROGRAM_BUILD_LOG, sizeof(buffer), &buffer, &len);
        printf("OpenCL Program Build Log : %s\n", buffer);

        cleanup();
        exit(EXIT_FAILURE);
    }

    //create OpenCL kernel
    oclKernel = clCreateKernel(oclProgram, "vecAdd", &ret_ocl);
    if(ret_ocl != CL_SUCCESS)
    {
        printf("OpenCL Error - clCreateKernel() Failed : %d. Exiting Now ...\n", ret_ocl);
        cleanup();
        exit(EXIT_FAILURE);
    }

    int size = inputLength * sizeof(cl_float);
    //allocate device memory 
    deviceInput1 = clCreateBuffer(oclContext, CL_MEM_READ_ONLY, size, NULL, &ret_ocl);
    if(ret_ocl != CL_SUCCESS)
    {
        printf("OpenCL Error - clCreateBuffer() Failed For 1st Array Input : %d. Exiting Now ...\n", ret_ocl);
        cleanup();
        exit(EXIT_FAILURE);
    }

    deviceInput2 = clCreateBuffer(oclContext, CL_MEM_READ_ONLY, size, NULL, &ret_ocl);
    if(ret_ocl != CL_SUCCESS)
    {
        printf("OpenCL Error - clCreateBuffer() Failed For 2nd Array Input : %d. Exiting Now ...\n", ret_ocl);
        cleanup();
        exit(EXIT_FAILURE);
    }

    deviceOutput = clCreateBuffer(oclContext, CL_MEM_WRITE_ONLY, size, NULL, &ret_ocl);
    if(ret_ocl != CL_SUCCESS)
    {
        printf("OpenCL Error - clCreateBuffer() Failed For Output Array : %d. Exiting Now ...\n", ret_ocl);
        cleanup();
        exit(EXIT_FAILURE);
    }

    //set up OpenCL kernel arguments. Our OpenCL kernel has 4 arguments 0, 1, 2, 3
    //set 1st argument i.e. deviceInput1
    ret_ocl = clSetKernelArg(oclKernel, 0, sizeof(cl_mem), (void *)&deviceInput1);
    if(ret_ocl != CL_SUCCESS)
    {
        printf("OpenCL Error - clSetKernelArg() Failed For 1st Argument : %d. Exiting Now ...\n", ret_ocl);
        cleanup();
        exit(EXIT_FAILURE);
    }

    //set 2nd Argument i.e deviceInput2
    ret_ocl = clSetKernelArg(oclKernel, 1, sizeof(cl_mem), (void *)&deviceInput2);
    if(ret_ocl != CL_SUCCESS)
    {
        printf("OpenCL Error - clSetKernelArg() Failed For 2nd Argument : %d. Exiting Now ...\n", ret_ocl);
        cleanup();
        exit(EXIT_FAILURE);
    }

    ret_ocl = clSetKernelArg(oclKernel, 2, sizeof(cl_mem), (void *)&deviceOutput);
    if(ret_ocl != CL_SUCCESS)
    {
        printf("OpenCL Error - clSetKernelArg() Failed For 3rd Argument : %d. Exiting Now ...\n", ret_ocl);
        cleanup();
        exit(EXIT_FAILURE);
    }

    ret_ocl = clSetKernelArg(oclKernel, 3, sizeof(cl_int), (void *)&inputLength);
    if(ret_ocl != CL_SUCCESS)
    {
        printf("OpenCL Error - clSetKernelArg() Failed For 4th Argument : %d. Exiting Now ...\n", ret_ocl);
        cleanup();
        exit(EXIT_FAILURE);
    }

    //write above input device buffer to device memory 
    ret_ocl = clEnqueueWriteBuffer(oclCommandQueue, deviceInput1, CL_FALSE, 0, size, hostInput1, 0, NULL, NULL);
    if(ret_ocl != CL_SUCCESS)
    {
        printf("OpenCL Error - clEnqueueWriteBuffer() Failed For 1st Input Device Buffer : %d. Exiting Now ...\n", ret_ocl);
        cleanup();
        exit(EXIT_FAILURE);
    }

    ret_ocl = clEnqueueWriteBuffer(oclCommandQueue, deviceInput2, CL_FALSE, 0, size, hostInput2, 0, NULL, NULL);
    if(ret_ocl != CL_SUCCESS)
    {
        printf("OpenCL Error - clEnqueueWriteBuffer() Failed For 2nd Input Device Buffer : %d. Exiting Now ...\n", ret_ocl);
        cleanup();
        exit(EXIT_FAILURE);
    }

    //run the kernel
    size_t global_size = 5;             //1-D Array size
    
    ret_ocl = clEnqueueNDRangeKernel(oclCommandQueue, oclKernel, 1, NULL, &global_size, NULL, 0, NULL, NULL);
    if(ret_ocl != CL_SUCCESS)
    {
        printf("OpenCL Error - clEnqueueNDRangeKernel() Failed : %d. Exiting Now ...\n", ret_ocl);
        cleanup();
        exit(EXIT_FAILURE);
    }

    //finish OpenCL command queue
    clFinish(oclCommandQueue);

    //read the output from the device to host
    ret_ocl = clEnqueueReadBuffer(oclCommandQueue, deviceOutput, CL_TRUE, 0, size, hostOutput, 0, NULL, NULL);
    if(ret_ocl != CL_SUCCESS)
    {
        printf("OpenCL Error - clEnqueueReadBuffer() Failed : %d. Exiting Now ...\n", ret_ocl);
        cleanup();
        exit(EXIT_FAILURE);
    }

    //results
    int i;
    for(i = 0; i < inputLength; i++)
    {
        printf("%f + %f = %f\n", hostInput1[i], hostInput2[i], hostOutput[i]);
    }

    //total cleanup
    cleanup();

    return (0);
}

void cleanup(void)
{
    //code

    //OpenCL cleanup
    if(oclKernel)
    {
        clReleaseKernel(oclKernel);
        oclKernel = NULL;
    }

    if(oclProgram)
    {
        clReleaseProgram(oclProgram);
        oclProgram = NULL;
    }

    if(oclCommandQueue)
    {
        clReleaseCommandQueue(oclCommandQueue);
        oclCommandQueue = NULL;
    }

    if(oclContext)
    {
        clReleaseContext(oclContext);
        oclContext = NULL;
    }

    //free allocated device memory
    if(deviceOutput)
    {
        clReleaseMemObject(deviceOutput);
        deviceOutput = NULL;
    }

    if(deviceInput2)
    {
        clReleaseMemObject(deviceInput2);
        deviceInput2 = NULL;
    }

    if(deviceInput1)
    {
        clReleaseMemObject(deviceInput1);
        deviceInput1 = NULL;
    }

    //free allocated host memory
    if(hostOutput)
    {
        free(hostOutput);
        hostOutput = NULL;
    }

    if(hostInput1)
    {
        free(hostInput1);
        hostInput1 = NULL;
    }

    if(hostInput2)
    {
        free(hostInput2);
        hostInput2 = NULL;
    }
}


