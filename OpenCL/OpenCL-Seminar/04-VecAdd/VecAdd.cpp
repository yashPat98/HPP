//headers
#include <stdio.h>
#include <stdlib.h>                     //exit()
#include <string.h>                     //strlen()
#include <math.h>                       //fabs()

#include <CL/opencl.h>                  //standard OpenCL header

#include "helper_timer.h"  

//global OpenCL variables
cl_int            ret_ocl;
cl_platform_id    oclPlatformID;        //compute platform id
cl_device_id      oclComputeDeviceID;   //compute device id
cl_context        oclContext;           //compute context
cl_command_queue  oclCommandQueue;      //compute command queue
cl_program        oclProgram;           //compute program
cl_kernel         oclKernel;            //compute kernel

char *oclSourceCode = NULL;
size_t sizeKernelCodeLength;

//odd number 11444777 is delibrate illustration (Nvidea OpenCL Samples)
int iNumberOfArrayElements = 11444777;
size_t localWorkSize = 256;
size_t globalWorkSize;

float *hostInput1   = NULL;
float *hostInput2   = NULL;
float *hostOutput   = NULL;
float *gold         = NULL;

cl_mem deviceInput1 = NULL;
cl_mem deviceInput2 = NULL;
cl_mem deviceOutput = NULL;

float timeOnCPU;
float timeOnGPU;

int main(void)
{
    //function declaration
    void fillArrayWithRandomNumbers(float *, int);
    size_t roundGlobalSizeToNearestMultipleOfLocalSize(int, unsigned int);
    void vecAddHost(const float *, const float *, float *, int);
    char* loadOclProgramSource(const char *, const char *, size_t *);
    void cleanup(void);

    //code
    //allocate host memory 
    hostInput1 = (float *)malloc(iNumberOfArrayElements * sizeof(float));
    if(hostInput1 == NULL)
    {
        printf("CPU Memory Fatal Error - Can Not Allocate Enough Memory For Host Input Array 1.\nExiting ...\n");
        cleanup();
        exit(EXIT_FAILURE);
    }

    hostInput2 = (float *)malloc(iNumberOfArrayElements * sizeof(float));
    if(hostInput2 == NULL)
    {
        printf("CPU Memory Fatal Error - Can Not Allocate Enough Memory For Host Input Array 2.\nExiting ...\n");
        cleanup();
        exit(EXIT_FAILURE);
    }

    hostOutput = (float *)malloc(iNumberOfArrayElements * sizeof(float));
    if(hostOutput == NULL)
    {
        printf("CPU Memory Fatal Error - Can Not Allocate Enough Memory For Host Output Array.\nExiting ...\n");
        cleanup();
        exit(EXIT_FAILURE);
    }

    gold = (float *)malloc(iNumberOfArrayElements * sizeof(float));
    if(gold == NULL)
    {
        printf("CPU Memory Fatal Error - Can Not Allocate Enough Memory For Gold Output Array.\nExiting ...\n");
        cleanup();
        exit(EXIT_FAILURE);
    }

    //fill above input host vectors with arbitary but hard coded data
    fillArrayWithRandomNumbers(hostInput1, iNumberOfArrayElements);
    fillArrayWithRandomNumbers(hostInput2, iNumberOfArrayElements);

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

    char gpu_name[255];
    clGetDeviceInfo(oclComputeDeviceID, CL_DEVICE_NAME, sizeof(gpu_name), &gpu_name, NULL);
    printf("GPU Device Name : %s\n", gpu_name);

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

    //create OpenCL program from .cl
    oclSourceCode = loadOclProgramSource("VecAdd.cl", "", &sizeKernelCodeLength);

    cl_int status = 0;
    oclProgram = clCreateProgramWithSource(oclContext, 1, (const char **)&oclSourceCode, &sizeKernelCodeLength, &ret_ocl);
    if(ret_ocl != CL_SUCCESS)
    {
        printf("OpenCL Error - clCreateProgramWithSource() Failed : %d. Exiting Now ...\n", ret_ocl);
        cleanup();
        exit(EXIT_FAILURE);
    }

    //build OpenCL program 
    ret_ocl = clBuildProgram(oclProgram, 0, NULL, NULL, NULL, NULL);
    if(ret_ocl != CL_SUCCESS)
    {
        printf("OpenCL Error - clBuildProgram() Failed : %d. Exiting Now ...\n", ret_ocl);

        size_t len;
        char buffer[2048];
        clGetProgramBuildInfo(oclProgram, oclComputeDeviceID, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
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

    int size = iNumberOfArrayElements * sizeof(cl_float);
    //allocate device memory
    deviceInput1 = clCreateBuffer(oclContext, CL_MEM_READ_ONLY, size, NULL, &ret_ocl);
    if(ret_ocl != CL_SUCCESS)
    {
        printf("OpenCL Error - clCreateBuffer() Failed For 1st Input Array : %d. Exiting Now ...\n", ret_ocl);
        cleanup();
        exit(EXIT_FAILURE);
    }

    deviceInput2 = clCreateBuffer(oclContext, CL_MEM_READ_ONLY, size, NULL, &ret_ocl);
    if(ret_ocl != CL_SUCCESS)
    {
        printf("OpenCL Error - clCreateBuffer() Failed For 2nd Input Array : %d. Exiting Now ...\n", ret_ocl);
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

    //set OpenCL kernel arguments
    //set 1st argument
    ret_ocl = clSetKernelArg(oclKernel, 0, sizeof(cl_mem), (void *)&deviceInput1);
    if(ret_ocl != CL_SUCCESS)
    {
        printf("OpenCL Error - clSetKernelArg() Failed For 1st Argument : %d. Exiting Now ...\n", ret_ocl);
        cleanup();
        exit(EXIT_FAILURE);
    }

    //set 2nd argument
    ret_ocl = clSetKernelArg(oclKernel, 1, sizeof(cl_mem), (void *)&deviceInput2);
    if(ret_ocl != CL_SUCCESS)
    {
        printf("OpenCL Error - clSetKernelArg() Failed For 2nd Argument : %d. Exiting Now ...\n", ret_ocl);
        cleanup();
        exit(EXIT_FAILURE);
    }

    //set 3rd argument
    ret_ocl = clSetKernelArg(oclKernel, 2, sizeof(cl_mem), (void *)&deviceOutput);
    if(ret_ocl != CL_SUCCESS)
    {
        printf("OpenCL Error - clSetKernelArg() Failed For 3rd Argument : %d. Exiting Now ...\n", ret_ocl);
        cleanup();
        exit(EXIT_FAILURE);
    }

    //set 4th argument
    ret_ocl = clSetKernelArg(oclKernel, 3, sizeof(cl_int), (void *)&iNumberOfArrayElements);
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
    globalWorkSize = roundGlobalSizeToNearestMultipleOfLocalSize(localWorkSize, iNumberOfArrayElements);

    //start timer 
    StopWatchInterface *timer = NULL;
    sdkCreateTimer(&timer);
    sdkStartTimer(&timer);

    ret_ocl = clEnqueueNDRangeKernel(oclCommandQueue, oclKernel, 1, NULL, &globalWorkSize, &localWorkSize, 0, NULL, NULL);
    if(ret_ocl != CL_SUCCESS)
    {
        printf("OpenCL Error - clEnqueueNDRangeKernel() Failed : %d. Exiting Now ...\n", ret_ocl);
        cleanup();
        exit(EXIT_FAILURE);
    }

    //finish OpenCL command queue
    clFinish(oclCommandQueue);

    //stop timer
    sdkStopTimer(&timer);
    timeOnGPU = sdkGetTimerValue(&timer);
    sdkDeleteTimer(&timer);
    timer = NULL;

    //read from device to host
    ret_ocl = clEnqueueReadBuffer(oclCommandQueue, deviceOutput, CL_TRUE, 0, size, hostOutput, 0, NULL, NULL);
    if(ret_ocl != CL_SUCCESS)
    {
        printf("OpenCL Error - clEnqueueReadBuffer() Failed : %d. Exiting Now ...\n", ret_ocl);
        cleanup();
        exit(EXIT_FAILURE);
    }

    vecAddHost(hostInput1, hostInput2, gold, iNumberOfArrayElements);

    //compare results for golden-host
    const float epsilon = 0.000001f;
    bool bAccuracy = true;
    int breakValue = 0;
    int i;
    for(i = 0; i < iNumberOfArrayElements; i++)
    {
        float val1 = gold[i];
        float val2 = hostOutput[i];
        
        if(fabs(val1 - val2) > epsilon)
        {
            bAccuracy = false;
            breakValue = i;
            break;
        }
    }

    if(bAccuracy == false)
    {
        printf("Break Value = %d\n", breakValue);
    }

    char str[125];
    if(bAccuracy == true)
        sprintf(str, "%s", "Comparison Of Output Arrays On CPU And GPU Are Accurate Within The Limit Of 0.000001");
    else
        sprintf(str, "%s", "Not All Comparison Of Output Arrays On CPU And GPU Are Accurate Within The Limit Of 0.000001");

    printf("\n1st Array Is From 0th Element %0.6f to %dth Element %0.6f\n", hostInput1[0], iNumberOfArrayElements - 1, hostInput1[iNumberOfArrayElements - 1]);
    printf("2nd Array Is From 0th Element %0.6f to %dth Element %0.6f\n", hostInput2[0], iNumberOfArrayElements - 1, hostInput2[iNumberOfArrayElements - 1]);
    printf("\nGlobal Work Size = %u And Local Work Size = %u\n", (unsigned int)globalWorkSize, (unsigned int)localWorkSize);
    printf("\nSum Of Each Element From Above 2 Arrays Creates 3rd Array As : \n");
    printf("3rd Array Is From 0th Element %0.6f to %dth Element %0.6f\n", hostOutput[0], iNumberOfArrayElements - 1, hostOutput[iNumberOfArrayElements - 1]);
    printf("\nThe Time Taken To Do Above Addition On CPU = %0.6f (ms)\n", timeOnCPU);
    printf("The Time Taken To Do Above Addition On GPU = %0.6f (ms)\n", timeOnGPU);
    printf("%s\n", str);

    //total cleanup
    cleanup();

    return (0);
}

void cleanup(void)
{
    //code
    //OpenCL cleanup
    if(oclSourceCode)
    {
        free((void *)oclSourceCode);
        oclSourceCode = NULL;
    }

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
    if(gold)
    {
        free(gold);
        gold = NULL;
    }

    if(hostOutput)
    {
        free(hostOutput);
        hostOutput = NULL;
    }

    if(hostInput2)
    {
        free(hostInput2);
        hostInput2 = NULL;
    }

    if(hostInput1)
    {
        free(hostInput1);
        hostInput1 = NULL;
    }
}

void fillArrayWithRandomNumbers(float *pFloatArray, int iSize)
{
    //code
    int i;
    const float fScale = 1.0f / (float)RAND_MAX;
    for(i = 0; i < iSize; i++)
    {
        pFloatArray[i] = fScale * rand();
    }
}

size_t roundGlobalSizeToNearestMultipleOfLocalSize(int local_size, unsigned int global_size)
{
    //code
    unsigned int r = global_size % local_size;
    
    if(r == 0)
        return (global_size);
    else
        return (global_size + local_size - r);
}

//Golden host vector addition function 
void vecAddHost(const float *in1, const float *in2, float *out, int iNumElements)
{
    int i;

    //start timer
    StopWatchInterface *timer = NULL;
    sdkCreateTimer(&timer);
    sdkStartTimer(&timer);

    for(i = 0; i < iNumElements; i++)
    {
        out[i] = in1[i] + in2[i];
    }

    //stop timer
    sdkStopTimer(&timer);
    timeOnCPU = sdkGetTimerValue(&timer);
    sdkDeleteTimer(&timer);
    timer = NULL;
}

char *loadOclProgramSource(const char *fileName, const char *preamble, size_t *sizeFinalLength)
{
    //variable declaration
    FILE *pFile = NULL;
    size_t sizeSourceLength;

    //code
    pFile = fopen(fileName, "rb");      //binary read
    if(pFile == NULL)
    {
        printf("loadOclProgramSource() Failed To Open The %s File.\nExiting Now ...\n", fileName);
        return (NULL);
    }

    size_t sizePreambleLength = (size_t)strlen(preamble);

    //get the length of source code
    fseek(pFile, 0, SEEK_END);
    sizeSourceLength = ftell(pFile);
    fseek(pFile, 0, SEEK_SET);

    //allocate buffer for source code string and read it in
    char *sourceString = (char *)malloc(sizeSourceLength + sizePreambleLength);
    if(sourceString == NULL)
    {
        printf("loadOclProgramSource() Failed To Allocate Memory For Source Code.\nExiting Now ...\n");
        fclose(pFile);
        pFile = NULL;

        return (NULL);
    }

    memcpy(sourceString, preamble, sizePreambleLength);
    if(fread((sourceString) + sizePreambleLength, sizeSourceLength, 1, pFile) != 1)
    {
        printf("loadOclProgramSource() Failed To Read From File.\nExiting Now ...\n");
        fclose(pFile);
        free(sourceString);
        pFile = NULL;
        sourceString = NULL;

        return (NULL);
    }

    //close the file and return the total length of the combined (preamble + source) string
    fclose(pFile);
    pFile = NULL;

    if(sizeFinalLength != NULL)
    {
        *sizeFinalLength = sizeSourceLength + sizePreambleLength;
    }
    sourceString[sizeSourceLength + sizePreambleLength] = '\0';

    return (sourceString);
}
