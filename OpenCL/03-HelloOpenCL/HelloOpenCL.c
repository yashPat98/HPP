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
    oclContext = clCreateContext(NULL, 1, &oclComputeDeviceID, NULL, NULL, &ret_ocl)
    if(ret_ocl != CL_SUCCESS)
    {
        printf("OpenCL Error - clCreateContext() Failed : %d. Exiting Now ...\n", ret_ocl);
        cleanup();
        error(EXIT_FAILURE);
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
    "__kernel void VecAdd(__global float *in1, __global float *in2, __global float *out, int len) \n" \ 
    "{ \n" \
        "int i = get_global_id(0); \n" \

        "if( i < len) \n" \
        "{ \n" \ 
            "out[i] = in1[i] + in2[i]; \n" \
        "} \n" \
    "} \n" \

    oclProgram = clCreateProgramWithSource(oclContext, 1, &oclKernelSource, NULL, &ret_ocl);
    if(ret_ocl != CL_SUCCESS)
    {
        printf("OpenCL Error - clCreateProgramWithSource() Failed : %d. Exiting Now ...\n", ret_ocl);

        size_t len;
        char buffer[2048];
        clGetProgramBuildInfo(oclProgram, oclComputeDeviceID, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
        printf("OpenCL Program Build Log : %s\n", buffer);

        cleanup();
        exit(EXIT_FAILURE);
    }

    //create OpenCL kernel
    oclKernel = clCreateKernel(oclProgram, "VecAdd", &ret_ocl);
    if(ret_ocl != CL_SUCCESS)
    {
        printf("OpenCL Error - clCreateKernel() Failed : %d. Exiting Now ...\n", ret_ocl);
        cleanup();
        exit(EXIT_FAILURE);
    }

    //todo
}
