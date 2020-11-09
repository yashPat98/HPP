// --- Headers ---
#include <cuda.h>
#include "../common/book.h"
#include "../common/cpu_anim.h"

// --- Macros ---
#define DIM       1024
#define PI        3.14159265f
#define MAX_TEMP  1.0f
#define MIN_TEMP  0.0001f
#define SPEED     0.25f

// --- Global Variables ---
struct DataBlock
{
    unsigned char   *output_bitmap;
    float           *device_inSrc;
    float           *device_outSrc;
    float           *device_constSrc;
    CPUAnimBitmap   *bitmap;
    
    cudaEvent_t      start, stop;
    float            totalTime;
    float            frames;
};

float *hostGrid = NULL;

texture<float, 2>  texConstSrc;
texture<float, 2>  texIn;
texture<float, 2>  texOut;

// --- CUDA KERNEL DEFINITION ---
__global__ void copy_const_kernel(float *input)
{
    //variable declaration
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int offset = x + y * blockDim.x * gridDim.x;

    float c = tex2D(texConstSrc, x, y);

    //code
    if(c != 0)
        input[offset] = c;
}

__global__ void blend_kernel(float *dst, bool dstOut)
{
    //variable declaration
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int offset = x + y * blockDim.x * gridDim.x;

    float t, l, c, r, b;
    
    //code
    //update the input temperature grid 
    if(dstOut)
    {
        t = tex2D(texIn, x, y-1);
        l = tex2D(texIn, x-1, y);
        c = tex2D(texIn, x, y);
        r = tex2D(texIn, x+1, y);
        b = tex2D(texIn, x, y+1);
    }
    else
    {
        t = tex2D(texOut, x, y-1);
        l = tex2D(texOut, x-1, y);
        c = tex2D(texOut, x, y);
        r = tex2D(texOut, x+1, y);
        b = tex2D(texOut, x, y+1);
    }

    dst[offset] = c + SPEED * (t + b + r + l - 4 * c);
}

// --- main() ---
int main(void)
{
    //function declaration
    void anim_gpu(DataBlock *d, int ticks);
    void cleanup(DataBlock *data);

    //variable declaration
    DataBlock data;
    CPUAnimBitmap bitmap(DIM, DIM, &data);
    cudaError_t err = cudaSuccess;

    //code
    data.bitmap     = &bitmap;
    data.totalTime  = 0.0f;
    data.frames     = 0.0f;

    //create cuda events 
    err = cudaEventCreate(&data.start);
    if(err != cudaSuccess)
    {
        printf("GPU Error - cudaEventCreate() failed : %s.\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaEventCreate(&data.stop);
    if(err != cudaSuccess)
    {
        printf("GPU Error - cudaEventCreate() failed : %s.\n", cudaGetErrorString(err));
        cudaEventDestroy(data.start);
        exit(EXIT_FAILURE);
    }

    //allocate device memory
    err = cudaMalloc((void **)&data.output_bitmap, bitmap.image_size());
    if(err != cudaSuccess)
    {
        printf("GPU Memory Fatal Error - cudaMalloc() failed : %s.\n", cudaGetErrorString(err));
        cleanup(&data);
        exit(EXIT_FAILURE);
    }

    err = cudaMalloc((void **)&data.device_inSrc, bitmap.image_size());
    if(err != cudaSuccess)
    {
        printf("GPU Memory Fatal Error - cudaMalloc() failed : %s.\n", cudaGetErrorString(err));
        cleanup(&data);
        exit(EXIT_FAILURE);
    }

    err = cudaMalloc((void **)&data.device_outSrc, bitmap.image_size());
    if(err != cudaSuccess)
    {
        printf("GPU Memory Fatal Error - cudaMalloc() failed : %s.\n", cudaGetErrorString(err));
        cleanup(&data);
        exit(EXIT_FAILURE);
    }
    
    err = cudaMalloc((void **)&data.device_constSrc, bitmap.image_size());
    if(err != cudaSuccess)
    {
        printf("GPU Memory Fatal Error - cudaMalloc() failed : %s.\n", cudaGetErrorString(err));
        cleanup(&data);
        exit(EXIT_FAILURE);
    }
    
    //bind textures
    cudaChannelFormatDesc desc = cudaCreateChannelDesc<float>();
    err = cudaBindTexture2D(NULL, texConstSrc, data.device_constSrc, desc, DIM, DIM, sizeof(float) * DIM);
    if(err != cudaSuccess)
    {
        printf("GPU Error - cudaBindTexture2D() failed : %s.\n", cudaGetErrorString(err));
        cleanup(&data);
        exit(EXIT_FAILURE);
    }

    err = cudaBindTexture2D(NULL, texIn, data.device_inSrc, desc, DIM, DIM, sizeof(float) * DIM);
    if(err != cudaSuccess)
    {
        printf("GPU Error - cudaBindTexture2D() failed : %s.\n", cudaGetErrorString(err));
        cleanup(&data);
        exit(EXIT_FAILURE);
    }

    err = cudaBindTexture2D(NULL, texOut, data.device_outSrc, desc, DIM, DIM, sizeof(float) * DIM);
    if(err != cudaSuccess)
    {
        printf("GPU Error - cudaBindTexture2D() failed : %s.\n", cudaGetErrorString(err));
        cleanup(&data);
        exit(EXIT_FAILURE);
    }

    //allocate host memory 
    hostGrid = (float *)malloc(bitmap.image_size());
    if(hostGrid == NULL)
    {
        printf("CPU Memory Fatal Error - can not allocate enough memory for grid.\n");
        cleanup(&data);
        exit(EXIT_FAILURE);
    }

    //initialize the constant data
    for(int i = 0; i < DIM * DIM; i++)
    {
        int x = i % DIM;
        int y = i / DIM;

        hostGrid[i] = 0;

        if((x > 300) && (x < 600) && (y > 310) && (y < 601))
            hostGrid[i] = MAX_TEMP;
    }

    hostGrid[DIM * 100 + 100] = (MAX_TEMP + MIN_TEMP) / 2.0f;
    hostGrid[DIM * 700 + 100] = MIN_TEMP;
    hostGrid[DIM * 300 + 300] = MIN_TEMP;
    hostGrid[DIM * 200 + 700] = MIN_TEMP;

    for(int y = 800; y < 900; y++)
    {
        for(int x = 400; x < 500; x++)
        {
            hostGrid[x + y * DIM] = MIN_TEMP;
        }
    }

    //copy the grid memory from host to device
    err = cudaMemcpy(data.device_constSrc, hostGrid, bitmap.image_size(), cudaMemcpyHostToDevice);
    if(err != cudaSuccess)
    {
        printf("GPU Memory Fatal Error - cudaMemcpy() failed : %s.\n", cudaGetErrorString(err));
        cleanup(&data);
        exit(EXIT_FAILURE);
    }

    //initialize the input data
    for(int y = 800; y < DIM; y++)
    {
        for(int x = 0; x < 200; x++)
        {
            hostGrid[x + y * DIM] = MAX_TEMP;
        }
    }

    err = cudaMemcpy(data.device_inSrc, hostGrid, bitmap.image_size(), cudaMemcpyHostToDevice);
    if(err != cudaSuccess)
    {
        printf("GPU Memory Fatal Error - cudaMemcpy() failed : %s.\n", cudaGetErrorString(err));
        cleanup(&data);
        exit(EXIT_FAILURE);
    }
    
    bitmap.anim_and_exit((void (*)(void *, int))anim_gpu, (void (*)(void *))cleanup);
}

void anim_gpu(DataBlock *data, int ticks)
{
    //function declaration
    void cleanup(DataBlock *data);

    //variable declaration
    CPUAnimBitmap *bitmap = data->bitmap;
    volatile bool dstOut = true;
    float elapsedTime = 0.0f;
    cudaError_t err = cudaSuccess;
    
    //code
    //start timer
    err = cudaEventRecord(data->start, 0);
    if(err != cudaSuccess)
    {
        printf("GPU Error - cudaEventRecord() failed : %s.\n", cudaGetErrorString(err));
        cleanup(data);
        exit(EXIT_FAILURE);
    }

    //kernel configuration
    dim3 DimGrid = dim3(DIM / 16, DIM / 16);
    dim3 DimBlock = dim3(16, 16);

    for(int i = 0; i < 90; i++)
    {
        float *in = NULL;
        float *out = NULL;

        if(dstOut)
        {
            in = data->device_inSrc;
            out = data->device_outSrc;
        }
        else
        {
            out = data->device_inSrc;
            in = data->device_outSrc;
        }

        copy_const_kernel<<<DimGrid, DimBlock>>>(in);
        
        blend_kernel<<<DimGrid, DimBlock>>>(out, dstOut);

        dstOut = !dstOut;
    }

    float_to_color<<<DimGrid, DimBlock>>>(data->output_bitmap, data->device_inSrc);

    //copy back to host memory
    err = cudaMemcpy(bitmap->get_ptr(), data->output_bitmap, bitmap->image_size(), cudaMemcpyDeviceToHost);
    if(err != cudaSuccess)
    {
        printf("GPU Memory Fatal Error - cudaMemcpy() failed : %s.\n", cudaGetErrorString(err));
        cleanup(data);
        exit(EXIT_FAILURE);
    }

    //stop timer
    err = cudaEventRecord(data->stop, 0);
    if(err != cudaSuccess)
    {
        printf("GPU Error - cudaEventRecord() failed : %s.\n", cudaGetErrorString(err));
        cleanup(data);
        exit(EXIT_FAILURE);
    }

    //synchronize
    err = cudaEventSynchronize(data->stop);
    if(err != cudaSuccess)
    {
        printf("GPU Error - cudaEventSynchronize() failed : %s.\n", cudaGetErrorString(err));
        cleanup(data);
        exit(EXIT_FAILURE);
    }

    err = cudaEventElapsedTime(&elapsedTime, data->start, data->stop);
    if(err != cudaSuccess)
    {
        printf("GPU Error - cudaEventElapsedTime() failed : %s.\n", cudaGetErrorString(err));
        cleanup(data);
        exit(EXIT_FAILURE);
    }

    data->totalTime += elapsedTime;
    data->frames = data->frames + 1;

    printf("Average Time Per Frame : %3.1f ms\n", data->totalTime / data->frames);
}

void cleanup(DataBlock *data)
{
    //code
    //free device memory
    if(data->device_inSrc)
    {
        cudaFree(data->device_inSrc);
        data->device_inSrc = NULL;
    }

    if(data->device_outSrc)
    {
        cudaFree(data->device_outSrc);
        data->device_outSrc = NULL;
    }

    if(data->device_constSrc)
    {
        cudaFree(data->device_constSrc);
        data->device_constSrc = NULL;
    }

    //free host memory 
    if(hostGrid)
    {
        free(hostGrid);
        hostGrid = NULL;
    }

    //unbind textures
    cudaUnbindTexture(texIn);
    cudaUnbindTexture(texOut);
    cudaUnbindTexture(texConstSrc);

    //deallocate events
    cudaEventDestroy(data->start);
    cudaEventDestroy(data->stop);
}
