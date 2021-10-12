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

size_t localWorkSize[2];
size_t globalWorkSize[2];

float *hostA = NULL;
float *hostB = NULL;
float *hostC = NULL;
float *CHost = NULL;

cl_mem deviceA = NULL;
cl_mem deviceB = NULL;
cl_mem deviceC = NULL;

float timeOnCPU;
float timeOnGPU;

int main(void)
{
    //function declaration
    void fillArrayWithRandomNumbers(float *, int);
    size_t roundGlobalSizeToNearestMultipleOfLocalSize(int, unsigned int);
    void matMulHost(float *, float *, float *, int, int, int);
    char* loadOclProgramSource(const char *, const char *, size_t *);
    void cleanup(void);

    //variable declaration
    int numARows;
    int numAColumns;
    int numBRows;
    int numBColumns;
    int numCRows;
    int numCColumns;
    int numCHostRows;
    int numCHostColumns;

    //code
    numARows = 4;
    numAColumns = 4;
    numBRows = 4;
    numBColumns = 4;

    numCRows = numARows;
    numCColumns = numBColumns;

    numCHostRows = numARows;
    numCHostColumns = numBColumns;

    int sizeA = numARows * numAColumns * sizeof(float);
    int sizeB = numBRows * numBColumns * sizeof(float);
    int sizeC = numCRows * numCColumns * sizeof(float);
    int sizeCHost = numCHostRows * numCHostColumns * sizeof(float);

    //allcate host memory
    hostA = (float *)malloc(sizeA);
    if(hostA == NULL)
    {
        printf("CPU Memory Fatal Error - Can Not Allocate Enough Memory For Host Input Matrix A.\nExiting Now ...\n");
        exit(EXIT_FAILURE);
    }

    hostB = (float *)malloc(sizeB);
    if(hostB == NULL)
    {
        printf("CPU Memory Fatal Error - Can Not Allocate Enough Memory For Host Input Matrix B.\nExiting Now ...\n");
        cleanup();
        exit(EXIT_FAILURE);        
    }

    hostC = (float *)malloc(sizeC);
    if(hostC == NULL)
    {
        printf("CPU Memory Fatal Error - Can Not Allocate Enough Memory For Host Output Matrix C.\nExiting Now ...\n");
        cleanup();
        exit(EXIT_FAILURE);        
    }

    CHost = (float *)malloc(sizeCHost);
    if(CHost == NULL)
    {
        printf("CPU Memory Fatal Error - Can Not Allocate Enough Memory For Host Output Matrix CHost.\nExiting Now ...\n");
        cleanup();
        exit(EXIT_FAILURE);        
    }

    //fill above input arrays with random float numbers
    fillArrayWithRandomNumbers(hostA, numARows * numAColumns);
    fillArrayWithRandomNumbers(hostB, numBRows * numBColumns);

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
    oclSourceCode = loadOclProgramSource("MatMul.cl", "", &sizeKernelCodeLength);

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
    oclKernel = clCreateKernel(oclProgram, "matrixMultiply", &ret_ocl);
    if(ret_ocl != CL_SUCCESS)
    {
        printf("OpenCL Error - clCreateKernel() Failed : %d. Exiting Now ...\n", ret_ocl);
        cleanup();
        exit(EXIT_FAILURE);
    }

    int size = (numCRows * numCColumns) * sizeof(cl_float);

    //allocate device memory
    deviceA = clCreateBuffer(oclContext, CL_MEM_READ_ONLY, size, NULL, &ret_ocl);
    if(ret_ocl != CL_SUCCESS)
    {
        printf("OpenCL Error - clCreateBuffer() Failed For 1st Input Array : %d. Exiting Now ...\n", ret_ocl);
        cleanup();
        exit(EXIT_FAILURE);
    }

    deviceB = clCreateBuffer(oclContext, CL_MEM_READ_ONLY, size, NULL, &ret_ocl);
    if(ret_ocl != CL_SUCCESS)
    {
        printf("OpenCL Error - clCreateBuffer() Failed For 2nd Input Array : %d. Exiting Now ...\n", ret_ocl);
        cleanup();
        exit(EXIT_FAILURE);
    }

    deviceC = clCreateBuffer(oclContext, CL_MEM_WRITE_ONLY, size, NULL, &ret_ocl);
    if(ret_ocl != CL_SUCCESS)
    {
        printf("OpenCL Error - clCreateBuffer() Failed For Output Array : %d. Exiting Now ...\n", ret_ocl);
        cleanup();
        exit(EXIT_FAILURE);
    }

    //set OpenCL kernel arguments
    //set 1st argument
    ret_ocl = clSetKernelArg(oclKernel, 0, sizeof(cl_mem), (void *)&deviceA);
    if(ret_ocl != CL_SUCCESS)
    {
        printf("OpenCL Error - clSetKernelArg() Failed For 1st Argument : %d. Exiting Now ...\n", ret_ocl);
        cleanup();
        exit(EXIT_FAILURE);
    }

    //set 2nd argument
    ret_ocl = clSetKernelArg(oclKernel, 1, sizeof(cl_mem), (void *)&deviceB);
    if(ret_ocl != CL_SUCCESS)
    {
        printf("OpenCL Error - clSetKernelArg() Failed For 2nd Argument : %d. Exiting Now ...\n", ret_ocl);
        cleanup();
        exit(EXIT_FAILURE);
    }

    //set 3rd argument
    ret_ocl = clSetKernelArg(oclKernel, 2, sizeof(cl_mem), (void *)&deviceC);
    if(ret_ocl != CL_SUCCESS)
    {
        printf("OpenCL Error - clSetKernelArg() Failed For 3rd Argument : %d. Exiting Now ...\n", ret_ocl);
        cleanup();
        exit(EXIT_FAILURE);
    }

    //set 4th argument
    ret_ocl = clSetKernelArg(oclKernel, 3, sizeof(cl_int), (void *)&numARows);
    if(ret_ocl != CL_SUCCESS)
    {
        printf("OpenCL Error - clSetKernelArg() Failed For 4th Argument : %d. Exiting Now ...\n", ret_ocl);
        cleanup();
        exit(EXIT_FAILURE);
    }

    //set 5th argument
    ret_ocl = clSetKernelArg(oclKernel, 4, sizeof(cl_int), (void *)&numAColumns);
    if(ret_ocl != CL_SUCCESS)
    {
        printf("OpenCL Error - clSetKernelArg() Failed For 5th Argument : %d. Exiting Now ...\n", ret_ocl);
        cleanup();
        exit(EXIT_FAILURE);
    }

    //set 6th argument
    ret_ocl = clSetKernelArg(oclKernel, 5, sizeof(cl_int), (void *)&numBRows);
    if(ret_ocl != CL_SUCCESS)
    {
        printf("OpenCL Error - clSetKernelArg() Failed For 6th Argument : %d. Exiting Now ...\n", ret_ocl);
        cleanup();
        exit(EXIT_FAILURE);
    }

    //set 7th argument
    ret_ocl = clSetKernelArg(oclKernel, 6, sizeof(cl_int), (void *)&numBColumns);
    if(ret_ocl != CL_SUCCESS)
    {
        printf("OpenCL Error - clSetKernelArg() Failed For 7th Argument : %d. Exiting Now ...\n", ret_ocl);
        cleanup();
        exit(EXIT_FAILURE);
    }

    //set 8th argument
    ret_ocl = clSetKernelArg(oclKernel, 7, sizeof(cl_int), (void *)&numCRows);
    if(ret_ocl != CL_SUCCESS)
    {
        printf("OpenCL Error - clSetKernelArg() Failed For 8th Argument : %d. Exiting Now ...\n", ret_ocl);
        cleanup();
        exit(EXIT_FAILURE);
    }

    //set 9th argument
    ret_ocl = clSetKernelArg(oclKernel, 8, sizeof(cl_int), (void *)&numCColumns);
    if(ret_ocl != CL_SUCCESS)
    {
        printf("OpenCL Error - clSetKernelArg() Failed For 9th Argument : %d. Exiting Now ...\n", ret_ocl);
        cleanup();
        exit(EXIT_FAILURE);
    }

    //write above input device buffer to device memory
    ret_ocl = clEnqueueWriteBuffer(oclCommandQueue, deviceA, CL_FALSE, 0, size, hostA, 0, NULL, NULL);
    if(ret_ocl != CL_SUCCESS)
    {
        printf("OpenCL Error - clEnqueueWriteBuffer() Failed For 1st Input Device Buffer : %d. Exiting Now ...\n", ret_ocl);
        cleanup();
        exit(EXIT_FAILURE);
    }

    ret_ocl = clEnqueueWriteBuffer(oclCommandQueue, deviceB, CL_FALSE, 0, size, hostB, 0, NULL, NULL);
    if(ret_ocl != CL_SUCCESS)
    {
        printf("OpenCL Error - clEnqueueWriteBuffer() Failed For 2nd Input Device Buffer : %d. Exiting Now ...\n", ret_ocl);
        cleanup();
        exit(EXIT_FAILURE);
    }

    //run the kernel
    localWorkSize[0] = 4;
    localWorkSize[1] = 4;

    globalWorkSize[0] = roundGlobalSizeToNearestMultipleOfLocalSize(localWorkSize[0], numCColumns);
    globalWorkSize[1] = roundGlobalSizeToNearestMultipleOfLocalSize(localWorkSize[1], numCRows);

    //start timer 
    StopWatchInterface *timer = NULL;
    sdkCreateTimer(&timer);
    sdkStartTimer(&timer);

    ret_ocl = clEnqueueNDRangeKernel(oclCommandQueue, oclKernel, 2, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
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
    ret_ocl = clEnqueueReadBuffer(oclCommandQueue, deviceC, CL_TRUE, 0, size, hostC, 0, NULL, NULL);
    if(ret_ocl != CL_SUCCESS)
    {
        printf("OpenCL Error - clEnqueueReadBuffer() Failed : %d. Exiting Now ...\n", ret_ocl);
        cleanup();
        exit(EXIT_FAILURE);
    }

    matMulHost(hostA, hostB, CHost, numAColumns, numCHostRows, numCHostColumns);

    //compare results for golden-host
    const float epsilon = 0.000001f;
    bool bAccuracy = true;
    int breakValue = 0;
    int i;
    for(i = 0; i < (numARows * numAColumns); i++)
    {
        float val1 = CHost[i];
        float val2 = hostC[i];
        
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

    printf("\n1st Matrix Is From 0th Element %0.6f to %dth Element %0.6f\n", hostA[0], (numARows * numAColumns) - 1, hostA[(numARows * numAColumns) - 1]);
    printf("2nd Matrix Is From 0th Element %0.6f to %dth Element %0.6f\n", hostB[0], (numBRows * numBColumns) - 1, hostB[(numBRows * numBColumns) - 1]);
    printf("\nGlobal Work Size = (%u, %u) And Local Work Size = (%u, %u)\n", (unsigned int)globalWorkSize[0], (unsigned int)globalWorkSize[1], (unsigned int)localWorkSize[0], (unsigned int)localWorkSize[1]);
    printf("\nMultiplication Of Above 2 Matrices Creates 3rd Matrix As : \n");
    printf("3rd Matrix Is From 0th Element %0.6f to %dth Element %0.6f\n", hostC[0], (numCRows * numCColumns) - 1, hostC[(numCRows * numCColumns) - 1]);
    printf("\nThe Time Taken To Do Above Calculations On CPU = %0.6f (ms)\n", timeOnCPU);
    printf("The Time Taken To Do Above Calculations On GPU = %0.6f (ms)\n", timeOnGPU);
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
    if(deviceA)
    {
        clReleaseMemObject(deviceA);
        deviceA = NULL;
    }

    if(deviceB)
    {
        clReleaseMemObject(deviceB);
        deviceB = NULL;
    }

    if(deviceC)
    {
        clReleaseMemObject(deviceC);
        deviceC = NULL;
    }

    //free allocated host memory
    if(CHost)
    {
        free(CHost);
        CHost = NULL;
    }

    if(hostC)
    {
        free(hostC);
        hostC = NULL;
    }

    if(hostB)
    {
        free(hostB);
        hostB = NULL;
    }

    if(hostA)
    {
        free(hostA);
        hostA = NULL;
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
void matMulHost(float *A, float *B, float *C, int iAColumns, int iCRows, int iCColumns)
{
    //start timer
    StopWatchInterface *timer = NULL;
    sdkCreateTimer(&timer);
    sdkStartTimer(&timer);

    for(int i = 0; i < iCRows; i++)
    {
        for(int j = 0; j < iCColumns; j++)
        {
            float sum = 0.0f;
            for(int k = 0; k < iAColumns; k++)
            {
                float a = A[i * iAColumns + k];
                float b = B[k * iCColumns + j];

                sum += a * b;
            }

            C[i * iCColumns + j] = sum;
        }
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

