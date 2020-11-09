// --- Headers ---
#include <cuda.h>
#include <stdio.h>

// --- Macros ---
#define CHUNK    (1024 * 1024)
#define SIZE     (CHUNK * 20)

// --- Variable Declaration ---
int *hostInputA = NULL;
int *hostInputB = NULL;
int *hostOutput = NULL;

int *deviceInputA0 = NULL;
int *deviceInputB0 = NULL;
int *deviceOutput0 = NULL;

int *deviceInputA1 = NULL;
int *deviceInputB1 = NULL;
int *deviceOutput1 = NULL;

cudaEvent_t start, stop;
cudaStream_t stream0, stream1;

// --- CUDA KERNEL DEFINITION ---
__global__ void kernel(int *inputA, int *inputB, int *output)
{
    //variable declaration
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    //code
    if(tid < CHUNK)
    {
        int tid1 = (tid + 1) % 256;
        int tid2 = (tid + 2) % 256;

        float inputA_avg = (inputA[tid] + inputA[tid1] + inputA[tid2]) / 3.0f;
        float inputB_avg = (inputB[tid] + inputB[tid1] + inputB[tid2]) / 3.0f;
        
        output[tid] = (inputA_avg + inputB_avg) / 2;
    }
}

// --- main() ---
int main(void)
{
    //function declaration
    void cleanup(void);

    //variable declaration
    cudaError_t err = cudaSuccess;
    cudaDeviceProp prop;
    int deviceID;
    float elapsedTime;
    
    //code
    //check for device overlap capability
    err = cudaGetDevice(&deviceID);
    if(err != cudaSuccess)
    {
        printf("GPU Error - cudaGetDevice() failed : %s.\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaGetDeviceProperties(&prop, deviceID);
    if(err != cudaSuccess)
    {
        printf("GPU Error - cudaGetDeviceProperties() failed : %s.\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    if(!prop.deviceOverlap)
    {
        printf("Device will not handle overlaps, so no speed up from streams\n");
        return (0);
    }

    //create cuda events
    err = cudaEventCreate(&start);
    if(err != cudaSuccess)
    {
        printf("GPU Error - cudaEventCreate() failed for start: %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaEventCreate(&stop);
    if(err != cudaSuccess)
    {
        printf("GPU Error - cudaEventCreate() failed for stop : %s\n", cudaGetErrorString(err));
        cudaEventDestroy(start);
        exit(EXIT_FAILURE);
    }

    //initialize the stream0 & stream1
    err = cudaStreamCreate(&stream0);
    if(err != cudaSuccess)
    {
        printf("GPU Error - cudaStreamCreate() failed for stream0 : %s\n", cudaGetErrorString(err));
        cleanup();
        exit(EXIT_FAILURE);
    }    

    err = cudaStreamCreate(&stream1);
    if(err != cudaSuccess)
    {
        printf("GPU Error - cudaStreamCreate() failed for stream1 : %s\n", cudaGetErrorString(err));
        cleanup();
        exit(EXIT_FAILURE);
    } 

    //allocate host memory (page-locked)
    err = cudaHostAlloc((void **)&hostInputA, SIZE * sizeof(int), cudaHostAllocDefault);
    if(err != cudaSuccess)
    {
        printf("CPU Memory Fatal Error - cudaHostAlloc() failed for hostInputA : %s.\n", cudaGetErrorString(err));
        cleanup();
        exit(EXIT_FAILURE);
    }

    err = cudaHostAlloc((void **)&hostInputB, SIZE * sizeof(int), cudaHostAllocDefault);
    if(err != cudaSuccess)
    {
        printf("CPU Memory Fatal Error - cudaHostAlloc() failed for hostInputB : %s.\n", cudaGetErrorString(err));
        cleanup();
        exit(EXIT_FAILURE);
    }

    err = cudaHostAlloc((void **)&hostOutput, SIZE * sizeof(int), cudaHostAllocDefault);
    if(err != cudaSuccess)
    {
        printf("CPU Memory Fatal Error - cudaHostAlloc() failed for hostOutput : %s.\n", cudaGetErrorString(err));
        cleanup();
        exit(EXIT_FAILURE);
    }

    //allocate device memory for stream 0
    err = cudaMalloc((void **)&deviceInputA0, CHUNK * sizeof(int));
    if(err != cudaSuccess)
    {
        printf("GPU Memory Fatal Error - cudaMalloc() failed for deviceInputA0 : %s.\n", cudaGetErrorString(err));
        cleanup();
        exit(EXIT_FAILURE);
    }

    err = cudaMalloc((void **)&deviceInputB0, CHUNK * sizeof(int));
    if(err != cudaSuccess)
    {
        printf("GPU Memory Fatal Error - cudaMalloc() failed for deviceInputB0 : %s.\n", cudaGetErrorString(err));
        cleanup();
        exit(EXIT_FAILURE);
    }    

    err = cudaMalloc((void **)&deviceOutput0, CHUNK * sizeof(int));
    if(err != cudaSuccess)
    {
        printf("GPU Memory Fatal Error - cudaMalloc() failed for deviceOutput0 : %s.\n", cudaGetErrorString(err));
        cleanup();
        exit(EXIT_FAILURE);
    }    

    //allocate device memory for stream 1
    err = cudaMalloc((void **)&deviceInputA1, CHUNK * sizeof(int));
    if(err != cudaSuccess)
    {
        printf("GPU Memory Fatal Error - cudaMalloc() failed for deviceInputA1 : %s.\n", cudaGetErrorString(err));
        cleanup();
        exit(EXIT_FAILURE);
    }

    err = cudaMalloc((void **)&deviceInputB1, CHUNK * sizeof(int));
    if(err != cudaSuccess)
    {
        printf("GPU Memory Fatal Error - cudaMalloc() failed for deviceInputB1 : %s.\n", cudaGetErrorString(err));
        cleanup();
        exit(EXIT_FAILURE);
    }    

    err = cudaMalloc((void **)&deviceOutput1, CHUNK * sizeof(int));
    if(err != cudaSuccess)
    {
        printf("GPU Memory Fatal Error - cudaMalloc() failed for deviceOutput1 : %s.\n", cudaGetErrorString(err));
        cleanup();
        exit(EXIT_FAILURE);
    }

    //fill the host input memory
    for(int i = 0; i < SIZE; i++)
    {
        hostInputA[i] = (float)((1.0f / (float)RAND_MAX) * rand());
        hostInputB[i] = (float)((1.0f / (float)RAND_MAX) * rand());
    }

    //start timer 
    err = cudaEventRecord(start, 0);
    if(err != cudaSuccess)
    {
        printf("GPU Error - cudaEventRecord() failed for start : %s.\n", cudaGetErrorString(err));
        cleanup();
        exit(EXIT_FAILURE);
    }

    //cuda kernel configuration
    dim3 DimGrid = dim3(CHUNK / 256, 1, 1);
    dim3 DimBlock = dim3(256, 1, 1);

    //now loop over full data, in bite-sized chunks
    for(int i = 0; i < SIZE; i += (CHUNK * 2))
    {
        //stream 0
        //copy the locked memory to the device, async
        err = cudaMemcpyAsync(deviceInputA0, hostInputA + i, CHUNK * sizeof(int), cudaMemcpyHostToDevice, stream0);
        if(err != cudaSuccess)
        {
            printf("GPU Error - cudaMemcpyAsync() failed for Input1 : %s.\n", cudaGetErrorString(err));
            cleanup();
            exit(EXIT_FAILURE);
        }

        err = cudaMemcpyAsync(deviceInputB0, hostInputB + i, CHUNK * sizeof(int), cudaMemcpyHostToDevice, stream0);
        if(err != cudaSuccess)
        {
            printf("GPU Error - cudaMemcpyAsync() failed for Input2 : %s.\n", cudaGetErrorString(err));
            cleanup();
            exit(EXIT_FAILURE);
        }

        kernel<<<DimGrid, DimBlock, 0, stream0>>>(deviceInputA0, deviceInputB0, deviceOutput0);

        //copy the data from device to locked memory
        err = cudaMemcpyAsync(hostOutput + i, deviceOutput0, CHUNK * sizeof(int), cudaMemcpyDeviceToHost, stream0);
        if(err != cudaSuccess)
        {
            printf("GPU Error - cudaMemcpyAsync() failed for Output : %s.\n", cudaGetErrorString(err));
            cleanup();
            exit(EXIT_FAILURE);
        }

        //stream 1
        //copy the locked memory to the device, async
        err = cudaMemcpyAsync(deviceInputA1, hostInputA + i + CHUNK, CHUNK * sizeof(int), cudaMemcpyHostToDevice, stream1);
        if(err != cudaSuccess)
        {
            printf("GPU Error - cudaMemcpyAsync() failed for Input1 : %s.\n", cudaGetErrorString(err));
            cleanup();
            exit(EXIT_FAILURE);
        }

        err = cudaMemcpyAsync(deviceInputB1, hostInputB + i + CHUNK, CHUNK * sizeof(int), cudaMemcpyHostToDevice, stream1);
        if(err != cudaSuccess)
        {
            printf("GPU Error - cudaMemcpyAsync() failed for Input2 : %s.\n", cudaGetErrorString(err));
            cleanup();
            exit(EXIT_FAILURE);
        }

        kernel<<<DimGrid, DimBlock, 0, stream1>>>(deviceInputA1, deviceInputB1, deviceOutput1);

        //copy the data from device to locked memory
        err = cudaMemcpyAsync(hostOutput + i + CHUNK, deviceOutput1, CHUNK * sizeof(int), cudaMemcpyDeviceToHost, stream1);
        if(err != cudaSuccess)
        {
            printf("GPU Error - cudaMemcpyAsync() failed for Output : %s.\n", cudaGetErrorString(err));
            cleanup();
            exit(EXIT_FAILURE);
        }
    }

    err = cudaStreamSynchronize(stream0);
    if(err != cudaSuccess)
    {
        printf("GPU Error - cudaStreamSynchronize() failed : %s.\n", cudaGetErrorString(err));
        cleanup();
        exit(EXIT_FAILURE);
    }

    err = cudaStreamSynchronize(stream1);
    if(err != cudaSuccess)
    {
        printf("GPU Error - cudaStreamSynchronize() failed : %s.\n", cudaGetErrorString(err));
        cleanup();
        exit(EXIT_FAILURE);
    }

    //stop timer
    err = cudaEventRecord(stop, 0);
    if(err != cudaSuccess)
    {
        printf("GPU Error - cudaEventRecord() failed for stop : %s.\n", cudaGetErrorString(err));
        cleanup();
        exit(EXIT_FAILURE);
    }

    err = cudaEventSynchronize(stop);
    if(err != cudaSuccess)
    {
        printf("GPU Error - cudaEventSynchronize() failed for stop : %s.\n", cudaGetErrorString(err));
        cleanup();
        exit(EXIT_FAILURE);
    }

    err = cudaEventElapsedTime(&elapsedTime, start, stop);
    if(err != cudaSuccess)
    {
        printf("GPU Error - cudaEventElapsedTime() failed : %s.\n", cudaGetErrorString(err));
        cleanup();
        exit(EXIT_FAILURE);
    }    

    printf("Time taken for double stream : %3.1f ms\n", elapsedTime);

    //total cleanup
    cleanup();

    return (0);
}

void cleanup(void)
{
    //code
    //free device memory for stream 1
    if(deviceOutput1)
    {
        cudaFree(deviceOutput1);
        deviceOutput1 = NULL;
    }

    if(deviceInputB1)
    {
        cudaFree(deviceInputB1);
        deviceInputB1 = NULL;
    }

    if(deviceInputA1)
    {
        cudaFree(deviceInputA1);
        deviceInputA1 = NULL;
    }

    //free device memory for stream 0
    if(deviceOutput0)
    {
        cudaFree(deviceOutput0);
        deviceOutput0 = NULL;
    }

    if(deviceInputB0)
    {
        cudaFree(deviceInputB0);
        deviceInputB0 = NULL;
    }

    if(deviceInputA0)
    {
        cudaFree(deviceInputA0);
        deviceInputA0 = NULL;
    }

    //free host memory 
    if(hostOutput)
    {
        cudaFreeHost(hostOutput);
        hostOutput = NULL;
    }

    if(hostInputB)
    {
        cudaFreeHost(hostInputB);
        hostInputB = NULL;
    }

    if(hostInputA)
    {
        cudaFreeHost(hostInputA);
        hostInputA = NULL;
    }

    //destroy stream
    cudaStreamDestroy(stream0);
    cudaStreamDestroy(stream1);

    //destroy events
    cudaEventDestroy(stop);
    cudaEventDestroy(start);
}
