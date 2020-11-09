// --- Headers ---
#include <cuda.h>
#include "../common/book.h"

#define SIZE (64 * 1024 * 1024)
#define FAILURE -1

//--- Variable Declaration ---
int *host = NULL;
int *device = NULL;

cudaEvent_t start, stop;
float elapsedTime;

// --- main() ---
int main(void)
{
    //function declaration
    float malloc_time(int size, bool up);
    float cudaHostAlloc_time(int size, bool up);

    //variable declaration
    float elapsedTime;
    float MB = (float)(100.0f * SIZE * sizeof(int) / (1024.0f * 1024.0f));

    //code
    //copy with malloc
    elapsedTime = malloc_time(SIZE, true);
    if(elapsedTime == FAILURE)
        exit(EXIT_FAILURE);

    printf("Time using malloc() : %3.1f ms\n", elapsedTime);
    printf("MB/s during copy up : %3.1f\n", MB/(elapsedTime / 1000));

    elapsedTime = malloc_time(SIZE, false);
    if(elapsedTime == FAILURE)
        exit(EXIT_FAILURE);

    printf("Time using malloc() : %3.1f ms\n", elapsedTime);
    printf("MB/s during copy down : %3.1f\n", MB/(elapsedTime / 1000));

    printf("\n\n");
    //copy with cudaHostAlloc
    elapsedTime = cudaHostAlloc_time(SIZE, true);
    if(elapsedTime == FAILURE)
        exit(EXIT_FAILURE);

    printf("Time using cudaHostAlloc() : %3.1f ms\n", elapsedTime);
    printf("MB/s during copy up : %3.1f\n", MB/(elapsedTime / 1000));

    elapsedTime = cudaHostAlloc_time(SIZE, false);
    if(elapsedTime == FAILURE)
        exit(EXIT_FAILURE);

    printf("Time using cudaHostAlloc() : %3.1f ms\n", elapsedTime);
    printf("MB/s during copy down : %3.1f\n", MB/(elapsedTime / 1000));

    return (0);
}

float malloc_time(int size, bool up)
{
    //function declaration
    void cleanup_malloc(void);

    //variable declaration
    cudaError_t err = cudaSuccess;
    float elapsedTime;

    //code
    //create cuda events
    err = cudaEventCreate(&start);
    if(err != cudaSuccess)
    {
        printf("GPU Error - cudaEventCreate() failed for start : %s.\n", cudaGetErrorString(err));
        return (FAILURE);
    }

    err = cudaEventCreate(&stop);
    if(err != cudaSuccess)
    {
        printf("GPU Error - cudaEventCreate() failed for stop : %s.\n", cudaGetErrorString(err));
        cudaEventDestroy(start);
        return (FAILURE);
    }

    //allocate host memory using malloc
    host = (int *)malloc(size * sizeof(int));
    if(host == NULL)
    {
        printf("CPU Memory Fatal Error - malloc() failed for host.\n");
        cleanup_malloc();
        return (FAILURE);
    }

    //allocate device memory 
    err = cudaMalloc((void **)&device, sizeof(int) * size);
    if(err != cudaSuccess)
    {
        printf("GPU Memory Fatal Error - cudaMalloc() failed for device : %s.\n", cudaGetErrorString(err));
        cleanup_malloc();
        return (FAILURE);
    }

    //start timer
    err = cudaEventRecord(start, 0);
    if(err != cudaSuccess)
    {
        printf("GPU Error - cudaEventRecord() failed for start : %s.\n", cudaGetErrorString(err));
        cleanup_malloc();
        return (FAILURE);
    }

    //copy
    for(int i = 0; i < 100; i++)
    {
        if(up)
        {
            err = cudaMemcpy(device, host, size * sizeof(int), cudaMemcpyHostToDevice);
            if(err != cudaSuccess)
            {
                printf("GPU Error - cudaMemcpy() failed for Host To Device : %s.\n", cudaGetErrorString(err));
                cleanup_malloc();
                return (FAILURE);
            }
        }
        else
        {
            err = cudaMemcpy(host, device, size * sizeof(int), cudaMemcpyDeviceToHost);
            if(err != cudaSuccess)
            {
                printf("GPU Error - cudaMemcpy() failed for Device To Host : %s.\n", cudaGetErrorString(err));
                cleanup_malloc();
                return (FAILURE);
            }
        }
    }

    //stop timer
    err = cudaEventRecord(stop, 0);
    if(err != cudaSuccess)
    {
        printf("GPU Error - cudaEventRecord() failed for stop : %s.\n", cudaGetErrorString(err));
        cleanup_malloc();
        return (FAILURE);
    }

    err = cudaEventSynchronize(stop);
    if(err != cudaSuccess)
    {
        printf("GPU Error - cudaEventSynchronize() failed : %s.\n", cudaGetErrorString(err));
        cleanup_malloc();
        return (FAILURE);
    }

    err = cudaEventElapsedTime(&elapsedTime, start, stop);
    if(err != cudaSuccess)
    {
        printf("GPU Error - cudaEventElapsedTime() failed : %s.\n", cudaGetErrorString(err));
        cleanup_malloc();
        return (FAILURE);
    }

    //total cleanup
    cleanup_malloc();

    return (elapsedTime);
}

float cudaHostAlloc_time(int size, bool up)
{
    //function declaration
    void cleanup_cudaHostAlloc(void);

    //variable declaration
    cudaError_t err = cudaSuccess;
    float elapsedTime;

    //code
    //create cuda events
    err = cudaEventCreate(&start);
    if(err != cudaSuccess)
    {
        printf("GPU Error - cudaEventCreate() failed for start : %s.\n", cudaGetErrorString(err));
        return (FAILURE);
    }

    err = cudaEventCreate(&stop);
    if(err != cudaSuccess)
    {
        printf("GPU Error - cudaEventCreate() failed for stop : %s.\n", cudaGetErrorString(err));
        cudaEventDestroy(start);
        return (FAILURE);
    }

    //allocate host memory using cudaHostAlloc
    err = cudaHostAlloc((void **)&host, size * sizeof(int), cudaHostAllocDefault);
    if(err != cudaSuccess)
    {
        printf("GPU Memory Fatal Error - cudaHostAlloc() failed for host : %s.\n", cudaGetErrorString(err));
        cleanup_cudaHostAlloc();
        return (FAILURE);
    }

    //allocate device memory 
    err = cudaMalloc((void **)&device, sizeof(int) * size);
    if(err != cudaSuccess)
    {
        printf("GPU Memory Fatal Error - cudaMalloc() failed for device : %s.\n", cudaGetErrorString(err));
        cleanup_cudaHostAlloc();
        return (FAILURE);
    }

    //start timer
    err = cudaEventRecord(start, 0);
    if(err != cudaSuccess)
    {
        printf("GPU Error - cudaEventRecord() failed for start : %s.\n", cudaGetErrorString(err));
        cleanup_cudaHostAlloc();
        return (FAILURE);
    }

    //copy
    for(int i = 0; i < 100; i++)
    {
        if(up)
        {
            err = cudaMemcpy(device, host, size * sizeof(int), cudaMemcpyHostToDevice);
            if(err != cudaSuccess)
            {
                printf("GPU Error - cudaMemcpy() failed for Host To Device : %s.\n", cudaGetErrorString(err));
                cleanup_cudaHostAlloc();
                return (FAILURE);
            }
        }
        else
        {
            err = cudaMemcpy(host, device, size * sizeof(int), cudaMemcpyDeviceToHost);
            if(err != cudaSuccess)
            {
                printf("GPU Error - cudaMemcpy() failed for Device To Host : %s.\n", cudaGetErrorString(err));
                cleanup_cudaHostAlloc();
                return (FAILURE);
            }
        }
    }

    //stop timer
    err = cudaEventRecord(stop, 0);
    if(err != cudaSuccess)
    {
        printf("GPU Error - cudaEventRecord() failed for stop : %s.\n", cudaGetErrorString(err));
        cleanup_cudaHostAlloc();
        return (FAILURE);
    }

    err = cudaEventSynchronize(stop);
    if(err != cudaSuccess)
    {
        printf("GPU Error - cudaEventSynchronize() failed : %s.\n", cudaGetErrorString(err));
        cleanup_cudaHostAlloc();
        return (FAILURE);
    }

    err = cudaEventElapsedTime(&elapsedTime, start, stop);
    if(err != cudaSuccess)
    {
        printf("GPU Error - cudaEventElapsedTime() failed : %s.\n", cudaGetErrorString(err));
        cleanup_cudaHostAlloc();
        return (FAILURE);
    }

    //total cleanup
    cleanup_cudaHostAlloc();

    return (elapsedTime);
}

void cleanup_malloc(void)
{
    //code
    if(device)
    {
        cudaFree(device);
        device = NULL;
    }

    if(host)
    {
        free(host);
        host = NULL;
    }

    cudaEventDestroy(stop);
    cudaEventDestroy(start);
}

void cleanup_cudaHostAlloc(void)
{
    //code
    if(device)
    {
        cudaFree(device);
        device = NULL;
    }

    if(host)
    {
        cudaFreeHost(host);
        host = NULL;
    }

    cudaEventDestroy(stop);
    cudaEventDestroy(start);
}
