// --- Headers ---
#include <cuda.h>
#include <stdio.h>

// --- Macros ---
#define CHUNK    (1024 * 1024)
#define SIZE     (CHUNK * 20)

// --- Variable Declaration ---
int *hostInput1 = NULL;
int *hostInput2 = NULL;
int *hostOutput = NULL;

int *deviceInput1 = NULL;
int *deviceInput2 = NULL;
int *deviceOutput = NULL;

cudaEvent_t start, stop;
cudaStream_t stream;

// --- CUDA KERNEL DEFINITION ---
__global__ void kernel(int *input1, int *input2, int *output)
{
    //variable declaration
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    //code
    if(tid < CHUNK)
    {
        int tid1 = (tid + 1) % 256;
        int tid2 = (tid + 2) % 256;

        float input1_avg = (input1[tid] + input1[tid1] + input1[tid2]) / 3.0f;
        float input2_avg = (input2[tid] + input2[tid1] + input2[tid2]) / 3.0f;
        
        output[tid] = (input1_avg + input2_avg) / 2;
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

    //initialize the stream
    err = cudaStreamCreate(&stream);
    if(err != cudaSuccess)
    {
        printf("GPU Error - cudaStreamCreate() failed for stream : %s\n", cudaGetErrorString(err));
        cleanup();
        exit(EXIT_FAILURE);
    }    

    //allocate host memory (page-locked)
    err = cudaHostAlloc((void **)&hostInput1, SIZE * sizeof(int), cudaHostAllocDefault);
    if(err != cudaSuccess)
    {
        printf("CPU Memory Fatal Error - cudaHostAlloc() failed for hostInput1 : %s.\n", cudaGetErrorString(err));
        cleanup();
        exit(EXIT_FAILURE);
    }

    err = cudaHostAlloc((void **)&hostInput2, SIZE * sizeof(int), cudaHostAllocDefault);
    if(err != cudaSuccess)
    {
        printf("CPU Memory Fatal Error - cudaHostAlloc() failed for hostInput2 : %s.\n", cudaGetErrorString(err));
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

    //allocate device memory 
    err = cudaMalloc((void **)&deviceInput1, CHUNK * sizeof(int));
    if(err != cudaSuccess)
    {
        printf("GPU Memory Fatal Error - cudaMalloc() failed for deviceInput1 : %s.\n", cudaGetErrorString(err));
        cleanup();
        exit(EXIT_FAILURE);
    }

    err = cudaMalloc((void **)&deviceInput2, CHUNK * sizeof(int));
    if(err != cudaSuccess)
    {
        printf("GPU Memory Fatal Error - cudaMalloc() failed for deviceInput2 : %s.\n", cudaGetErrorString(err));
        cleanup();
        exit(EXIT_FAILURE);
    }    

    err = cudaMalloc((void **)&deviceOutput, CHUNK * sizeof(int));
    if(err != cudaSuccess)
    {
        printf("GPU Memory Fatal Error - cudaMalloc() failed for deviceOutput : %s.\n", cudaGetErrorString(err));
        cleanup();
        exit(EXIT_FAILURE);
    }    

    //fill the host input memory
    for(int i = 0; i < SIZE; i++)
    {
        hostInput1[i] = (float)((1.0f / (float)RAND_MAX) * rand());
        hostInput2[i] = (float)((1.0f / (float)RAND_MAX) * rand());
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
    for(int i = 0; i < SIZE; i += CHUNK)
    {
        //copy the locked memory to the device, async
        err = cudaMemcpyAsync(deviceInput1, hostInput1 + i, CHUNK * sizeof(int), cudaMemcpyHostToDevice, stream);
        if(err != cudaSuccess)
        {
            printf("GPU Error - cudaMemcpyAsync() failed for Input1 : %s.\n", cudaGetErrorString(err));
            cleanup();
            exit(EXIT_FAILURE);
        }

        err = cudaMemcpyAsync(deviceInput2, hostInput2 + i, CHUNK * sizeof(int), cudaMemcpyHostToDevice, stream);
        if(err != cudaSuccess)
        {
            printf("GPU Error - cudaMemcpyAsync() failed for Input2 : %s.\n", cudaGetErrorString(err));
            cleanup();
            exit(EXIT_FAILURE);
        }

        kernel<<<DimGrid, DimBlock, 0, stream>>>(deviceInput1, deviceInput2, deviceOutput);

        //copy the data from device to locked memory
        err = cudaMemcpyAsync(hostOutput + i, deviceOutput, CHUNK * sizeof(int), cudaMemcpyDeviceToHost, stream);
        if(err != cudaSuccess)
        {
            printf("GPU Error - cudaMemcpyAsync() failed for Output : %s.\n", cudaGetErrorString(err));
            cleanup();
            exit(EXIT_FAILURE);
        }
    }

    err = cudaStreamSynchronize(stream);
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

    printf("Time taken for single stream : %3.1f ms\n", elapsedTime);

    //total cleanup
    cleanup();

    return (0);
}

void cleanup(void)
{
    //code
    //free device memory 
    if(deviceOutput)
    {
        cudaFree(deviceOutput);
        deviceOutput = NULL;
    }

    if(deviceInput2)
    {
        cudaFree(deviceInput2);
        deviceInput2 = NULL;
    }

    if(deviceInput1)
    {
        cudaFree(deviceInput1);
        deviceInput1 = NULL;
    }

    //free host memory 
    if(hostOutput)
    {
        cudaFreeHost(hostOutput);
        hostOutput = NULL;
    }

    if(hostInput2)
    {
        cudaFreeHost(hostInput2);
        hostInput2 = NULL;
    }

    if(hostInput1)
    {
        cudaFreeHost(hostInput1);
        hostInput1 = NULL;
    }

    //destroy stream
    cudaStreamDestroy(stream);

    //destroy events
    cudaEventDestroy(stop);
    cudaEventDestroy(start);
}
