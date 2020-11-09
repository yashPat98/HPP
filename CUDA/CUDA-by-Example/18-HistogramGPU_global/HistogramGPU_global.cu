// --- Headers ---
#include "../common/book.h"

#define SIZE (100 * 1024 * 1024)

// --- Variable Declaration ---
unsigned char *hostData        = NULL;
unsigned int *hostHistogram    = NULL;

unsigned char *deviceData      = NULL;
unsigned int *deviceHistogram  = NULL;

cudaEvent_t start, stop;

// --- CUDA KERNEL DEFINITION ---
__global__ void HistogramKernel(unsigned char *data, long size, unsigned int *histogram)
{
    //variable declaration
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    //code 
    while(i < size)
    {
        atomicAdd(&histogram[data[i]], 1);
        i += stride;
    }
}

// --- main() ---
int main(void)
{
    //function declaration
    void cleanup(void);

    //variable declaration
    long histogramCount = 0;
    float elapsedTime;

    cudaDeviceProp prop;
    cudaError_t err = cudaSuccess;
    
    //code
    //create cuda events
    err = cudaEventCreate(&start);
    if(err != cudaSuccess)
    {
        printf("GPU Error - cudaEventCreate() failed for start : %s.\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    } 

    err = cudaEventCreate(&stop);
    if(err != cudaSuccess)
    {
        printf("GPU Error - cudaEventCreate() failed for stop : %s.\n", cudaGetErrorString(err));
        cudaEventDestroy(start);
        exit(EXIT_FAILURE);
    }

    //allocate host memory
    hostData = (unsigned char *)big_random_block(SIZE);
    if(hostData == NULL)
    {
        printf("CPU Memory Fatal Error - malloc() failed for hostData.\n");
        cleanup();
        exit(EXIT_FAILURE);
    }

    hostHistogram = (unsigned int *)malloc(256 * sizeof(float));
    if(hostHistogram == NULL)
    {
        printf("CPU Memory Fatal Error - malloc() failed for hostHistogram.\n");
        cleanup();
        exit(EXIT_FAILURE);
    }

    //allocate device memory 
    err = cudaMalloc((void **)&deviceData, SIZE);
    if(err != cudaSuccess)
    {
        printf("GPU Memory Fatal Error - cudaMalloc() failed for deviceData : %s.\n", cudaGetErrorString(err));
        cleanup();
        exit(EXIT_FAILURE);
    }

    err = cudaMalloc((void **)&deviceHistogram, 256 * sizeof(float));
    if(err != cudaSuccess)
    {
        printf("GPU Memory Fatal Error - cudaMalloc() failed for deviceHistogram : %s.\n", cudaGetErrorString(err));
        cleanup();
        exit(EXIT_FAILURE);
    }

    //zero-out host histogram memory
    for(int i = 0; i < 256; i++)
        hostHistogram[i] = 0;

    //zero-out device histogram memory
    err = cudaMemset(deviceHistogram, 0, 256 * sizeof(float));
    if(err != cudaSuccess)
    {
        printf("GPU Error - cudaMemset() failed for deviceHistogram : %s.\n", cudaGetErrorString(err));
        cleanup();
        exit(EXIT_FAILURE);
    }

    //copy data from host to device
    err = cudaMemcpy(deviceData, hostData, SIZE, cudaMemcpyHostToDevice);
    if(err != cudaSuccess)
    {
        printf("GPU Error - cudaMemcpy() failed for host to device : %s.\n", cudaGetErrorString(err));
        cleanup();
        exit(EXIT_FAILURE);
    }



    // --- KERNEL CONFIGURATION ---
    //kernel launch - 2x the number of multi processors gave best timing
    err = cudaGetDeviceProperties(&prop, 0);
    if(err != cudaSuccess)
    {
        printf("GPU Error - cudaDeviceProperties() : %s.\n", cudaGetErrorString(err));
        cleanup();
        exit(EXIT_FAILURE);
    }  

    int blocks = prop.multiProcessorCount;

    //start the timer
    err = cudaEventRecord(start, 0);
    if(err != cudaSuccess)
    {
        printf("GPU Error - cudaEventRecord() failed for start : %s.\n", cudaGetErrorString(err));
        cleanup();
        exit(EXIT_FAILURE);
    }

    //kernel launch
    HistogramKernel<<<blocks * 2, 256>>>(deviceData, SIZE, deviceHistogram);

    //copy histogram from device to host
    err = cudaMemcpy(hostHistogram, deviceHistogram, 256 * sizeof(float), cudaMemcpyDeviceToHost);
    if(err != cudaSuccess)
    {
        printf("GPU Error - cudaMemcpy() failed for device to host : %s.\n", cudaGetErrorString(err));
        cleanup();
        exit(EXIT_FAILURE);
    }

    //stop the timer
    err = cudaEventRecord(stop, 0);
    if(err != cudaSuccess)
    {
        printf("GPU Error - cudaEventRecord() failed for stop : %s.\n", cudaGetErrorString(err));
        cleanup();
        exit(EXIT_FAILURE);
    }    

    //sync to stop event
    err = cudaEventSynchronize(stop);
    if(err != cudaSuccess)
    {
        printf("GPU Error - cudaEventSynchronize() failed for stop : %s.\n", cudaGetErrorString(err));
        cleanup();
        exit(EXIT_FAILURE);
    }

    //calculate time
    err = cudaEventElapsedTime(&elapsedTime, start, stop);
    if(err != cudaSuccess)
    {
        printf("GPU Error - cudaEventElapsedTime() failed : %s.\n", cudaGetErrorString(err));
        cleanup();
        exit(EXIT_FAILURE);
    }

    printf("Time to generate : %3.1f ms\n", elapsedTime);

    for(int i = 0; i < 256; i++)
        histogramCount += hostHistogram[i];

    printf("Histogram Sum : %ld\n", histogramCount);

    //verify that we have same count on CPU
    for(int i = 0; i < SIZE; i++)
        hostHistogram[hostData[i]]--;
    
    for(int i = 0; i < 256; i++)
        if(hostHistogram[i] != 0)
            printf("Failure at %d ! Off by %d\n", i, hostHistogram[i]);

    //total cleanup
    cleanup();

    return (0);
}

void cleanup(void)
{
    //code
    if(deviceHistogram)
    {
        cudaFree(deviceHistogram);
        deviceHistogram = NULL;
    }

    if(deviceData)
    {
        cudaFree(deviceData);
        deviceData = NULL;
    }

    if(hostHistogram)
    {
        free(hostHistogram);
        hostHistogram = NULL;
    }

    if(hostData)
    {
        free(hostData);
        hostData = NULL;
    }

    cudaEventDestroy(stop);
    cudaEventDestroy(start);
}
