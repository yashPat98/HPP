//headers
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

#define N (33 * 1024)

//global variables declaration
int hostA[N];
int hostB[N];
int hostC[N];

int *deviceA;
int *deviceB;
int *deviceC;

// *** CUDA KERNEL DEFINITION ***
__global__ void add(int *a, int *b, int *c)
{
    //variable declaration
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    //code
    while(tid < N)
    {
        c[tid] = a[tid] + b[tid];
        tid += blockDim.x * gridDim.x;
    }
}

int main(int argc, char *argv[])
{
    //function declaration
    void cleanup(void);

    //code
    cudaError_t err = cudaSuccess;
    
    //allocate memory on device
    err = cudaMalloc((void **)&deviceA, N * sizeof(int));
    if(err != cudaSuccess)
    {
        printf("GPU Memory Fatal Error = %s In File Name %s At Line No %d.\nExiting Now ...\n", cudaGetErrorString(err), __FILE__, __LINE__);
        exit(EXIT_FAILURE);
    }

    err = cudaMalloc((void **)&deviceB, N * sizeof(int));
    if(err != cudaSuccess)
    {
        printf("GPU Memory Fatal Error = %s In File Name %s At Line No %d.\nExiting Now ...\n", cudaGetErrorString(err), __FILE__, __LINE__);
        cleanup();
        exit(EXIT_FAILURE);
    }

    err = cudaMalloc((void **)&deviceC, N * sizeof(int));
    if(err != cudaSuccess)
    {
        printf("GPU Memory Fatal Error = %s In File Name %s At Line No %d.\nExiting Now ...\n", cudaGetErrorString(err), __FILE__, __LINE__);
        cleanup();
        exit(EXIT_FAILURE);
    }

    //fill the host input array
    for(int i = 0; i < N; i++)
    {
        hostA[i] = i;
        hostB[i] = i * i;
    }

    //copy the host input array to device memory
    err = cudaMemcpy(deviceA, hostA, (N * sizeof(int)), cudaMemcpyHostToDevice);
    if(err != cudaSuccess)
    {
        printf("GPU Memory Fatal Error = %s In File Name %s At Line No %d.\nExiting Now ...\n", cudaGetErrorString(err), __FILE__, __LINE__);
        cleanup();
        exit(EXIT_FAILURE);
    }

    err = cudaMemcpy(deviceB, hostB, (N * sizeof(int)), cudaMemcpyHostToDevice);
    if(err != cudaSuccess)
    {
        printf("GPU Memory Fatal Error = %s In File Name %s At Line No %d.\nExiting Now ...\n", cudaGetErrorString(err), __FILE__, __LINE__);
        cleanup();
        exit(EXIT_FAILURE);
    }
    
    //cuda kernel configuration
    dim3 DimGrid = 128;
    dim3 DimBlock = 128;

    add<<<DimGrid, DimBlock>>>(deviceA, deviceB, deviceC);

    //copy output array back to host 
    err = cudaMemcpy(hostC, deviceC, (N * sizeof(int)), cudaMemcpyDeviceToHost);
    if(err != cudaSuccess)
    {
        printf("GPU Memory Fatal Error = %s In File Name %s At Line No %d.\nExiting Now ...\n", cudaGetErrorString(err), __FILE__, __LINE__);
        cleanup();
        exit(EXIT_FAILURE);
    }    

    //verify the results 
    bool success = true;
    for(int i = 0; i < N; i++)
    {
        if(hostA[i] + hostB[i] != hostC[i])
        {
            printf("Error : %d + %d != %d\n", hostA[i], hostB[i], hostC[i]);
            success = false;
        }
    }

    if(success)
    {
        printf("Addition Is Successful On GPU !\n");
    }

    //total cleanup
    cleanup();

    return (0);
}

void cleanup(void)
{
    if(deviceA)
    {
        cudaFree(deviceA);
        deviceA = NULL;
    }

    if(deviceB)
    {
        cudaFree(deviceB);
        deviceB = NULL;
    }

    if(deviceC)
    {
        cudaFree(deviceC);
        deviceC = NULL;
    }
}
