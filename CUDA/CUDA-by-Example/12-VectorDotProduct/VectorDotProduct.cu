//headers
#include <stdio.h>
#include <cuda.h>

#define imin(a, b)       ((a < b) ? a : b)
#define sum_squares(x)   (x * (x + 1) * (2 * x + 1) / 6)

//global variables
float *hostA = NULL;
float *hostB = NULL;
float *partial_hostC = NULL;

float *deviceA = NULL;
float *deviceB = NULL;
float *partial_deviceC = NULL;

const int iNumberOfArrayElements = 33 * 1024;
const int threadsPerBlock = 256;
const int blocksPerGrid = imin(32, (iNumberOfArrayElements + threadsPerBlock - 1) / threadsPerBlock);

// *** CUDA KERNEL DEFINITION ***
__global__ void vecDotProduct(float *input1, float *input2, float *output)
{
    //variable declaration
    //shared across all threads within block
    __shared__ float cache[threadsPerBlock];                           

    int tid        = blockIdx.x * blockDim.x + threadIdx.x;          
    int cacheIndex = threadIdx.x;
    float temp     = 0;

    //code
    while(tid < iNumberOfArrayElements)
    {
        temp += input1[tid] * input2[tid];
        tid += blockDim.x * gridDim.x;
    }

    //set the cache values
    cache[cacheIndex] = temp;

    //synchronize threads in the block
    __syncthreads();

    //summation reduction
    int i = blockDim.x / 2;
    while(i != 0)
    {
        if(cacheIndex < i)
            cache[cacheIndex] += cache[cacheIndex + i];

        __syncthreads();
        i /= 2;
    }

    //copy to output memory
    if(cacheIndex == 0)
        output[blockIdx.x] = cache[0];
}

int main(void)
{
    //function declaration
    void cleanup(void);

    //code
    //allocate memory on host 
    hostA = (float *)malloc(iNumberOfArrayElements * sizeof(float));
    if(hostA == NULL)
    {
        printf("CPU Memory Fatal Error - Can Not Allocate Enough Memory For Host Input Array 1.\nExiting Now ...\n");
        exit(EXIT_FAILURE);
    }

    hostB = (float *)malloc(iNumberOfArrayElements * sizeof(float));
    if(hostB == NULL)
    {
        printf("CPU Memory Fatal Error - Can Not Allocate Enough Memory For Host Input Array 2.\nExiting Now ...\n");
        cleanup();
        exit(EXIT_FAILURE);
    }

    partial_hostC = (float *)malloc(blocksPerGrid * sizeof(float));
    if(partial_hostC == NULL)
    {
        printf("CPU Memory Fatal Error - Can Not Allocate Enough Memory For Host Output Array.\nExiting Now ...\n");
        cleanup();
        exit(EXIT_FAILURE);
    }

    //allocate memory on device
    cudaError_t err = cudaSuccess;
    err = cudaMalloc((void **)&deviceA, iNumberOfArrayElements * sizeof(float));
    if(err != cudaSuccess)
    {
        printf("GPU Memory Fatal Error - %s In The File Name %s At Line No %d.\nExiting Now ...\n", cudaGetErrorString(err), __FILE__, __LINE__);
        cleanup();
        exit(EXIT_FAILURE);
    }    

    err = cudaMalloc((void **)&deviceB, iNumberOfArrayElements * sizeof(float));
    if(err != cudaSuccess)
    {
        printf("GPU Memory Fatal Error - %s In The File Name %s At Line No %d.\nExiting Now ...\n", cudaGetErrorString(err), __FILE__, __LINE__);
        cleanup();
        exit(EXIT_FAILURE);
    }    

    err = cudaMalloc((void **)&partial_deviceC, blocksPerGrid * sizeof(float));
    if(err != cudaSuccess)
    {
        printf("GPU Memory Fatal Error - %s In The File Name %s At Line No %d.\nExiting Now ...\n", cudaGetErrorString(err), __FILE__, __LINE__);
        cleanup();
        exit(EXIT_FAILURE);
    } 

    //fill the host input array 
    for(int i = 0; i < iNumberOfArrayElements; i++)
    {
        hostA[i] = i;
        hostB[i] = i * 2;
    }

    //copy the host input array to device input
    err = cudaMemcpy(deviceA, hostA, iNumberOfArrayElements * sizeof(float), cudaMemcpyHostToDevice);
    if(err != cudaSuccess)
    {
        printf("GPU Memory Fatal Error - %s In The File Name %s At Line No %d.\nExiting Now ...\n", cudaGetErrorString(err), __FILE__, __LINE__);
        cleanup();
        exit(EXIT_FAILURE);
    } 

    err = cudaMemcpy(deviceB, hostB, iNumberOfArrayElements * sizeof(float), cudaMemcpyHostToDevice);
    if(err != cudaSuccess)
    {
        printf("GPU Memory Fatal Error - %s In The File Name %s At Line No %d.\nExiting Now ...\n", cudaGetErrorString(err), __FILE__, __LINE__);
        cleanup();
        exit(EXIT_FAILURE);
    }

    //launch the kernel
    vecDotProduct<<<blocksPerGrid, threadsPerBlock>>>(deviceA, deviceB, partial_deviceC);

    //copy the device output array back to host
    err = cudaMemcpy(partial_hostC, partial_deviceC, blocksPerGrid * sizeof(float), cudaMemcpyDeviceToHost);
    if(err != cudaSuccess)
    {
        printf("GPU Memory Fatal Error - %s In The File Name %s At Line No %d.\nExiting Now ...\n", cudaGetErrorString(err), __FILE__, __LINE__);
        cleanup();
        exit(EXIT_FAILURE);
    }

    //calculate final result on host
    float finalC = 0.0f;
    for(int i = 0; i < blocksPerGrid; i++)
    {
        finalC += partial_hostC[i];
    }

    //check if the final value is correct
    if(finalC == (2 * sum_squares((float)(iNumberOfArrayElements - 1))))
        printf("Dot Product Calculated On Device Is Accurate.\n");
    else
        printf("Dot Product Calculated On Device Is Not Accurate.\n");

    printf("Dot Product = %0.6.\n", finalC);
    printf("Expected Product = %0.6f.\n", 2 * sum_squares((float)(iNumberOfArrayElements - 1)));

    //total cleanup
    cleanup();
    
    return (0);
}

void cleanup(void)
{
    //code
    //free device memory 
    if(partial_deviceC)
    {
        cudaFree(partial_deviceC);
        partial_deviceC = NULL;
    }

    if(deviceB)
    {
        cudaFree(deviceB);
        deviceB = NULL;
    }

    if(deviceA)
    {
        cudaFree(deviceA);
        deviceA = NULL;
    }

    //free host memory
    if(partial_hostC)
    {
        free(partial_hostC);
        partial_hostC = NULL;
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
