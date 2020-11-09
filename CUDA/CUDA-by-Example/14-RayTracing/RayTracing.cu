//headers
#include <cuda.h>
#include "../common/book.h"
#include "../common/cpu_bitmap.h"

//macros
#define DIM       1024
#define MAX       20
#define rnd(x)   (x * rand() / RAND_MAX)
#define INF       2e10f

struct SPHERE
{
    float x, y, z;
    float r, g, b;
    float radius;

    __device__ float hit(float ox, float oy, float *n)
    {
        float dx = ox - x;
        float dy = oy - y;

        if((dx * dx + dy * dy) < (radius * radius))
        {
            float dz = sqrtf(radius * radius - dx * dx - dy * dy);
            *n = dz / sqrtf(radius * radius);
            return (dz + z);
        }

        return (-INF);
    }
};

__constant__ SPHERE spheres[MAX];
SPHERE *temp = NULL;

unsigned char *device = NULL;
cudaEvent_t start, stop;
float elapsedTime;

// --- CUDA KERNEL DEFINITION ---
__global__ void kernel(unsigned char *ptr)
{
    //variable declaration
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int offset = x + y * blockDim.x * gridDim.x;

    float ox = (x - DIM / 2);
    float oy = (y - DIM / 2);

    float r = 0.0f;
    float g = 0.0f;
    float b = 0.0f;

    float maxz = -INF;
    
    //code
    for(int i = 0; i < MAX; i++)
    {
        float n;
        float t = spheres[i].hit(ox, oy, &n);

        if(t > maxz)
        {
            float fScale = n;
            r = spheres[i].r * fScale;
            g = spheres[i].g * fScale;
            b = spheres[i].b * fScale;

            maxz = t;
        }
    }

    ptr[offset * 4 + 0] = (int)(r * 255);
    ptr[offset * 4 + 1] = (int)(g * 255);
    ptr[offset * 4 + 2] = (int)(b * 255);
    ptr[offset * 4 + 3] = 255;
}

struct DataBlock
{
    unsigned char *dev_bitmap;
};

int main(void)
{
    //function declaration
    void cleanup(void);

    //variable declaration
    DataBlock data;
    CPUBitmap bitmap(DIM, DIM, &data);
    
    cudaError_t err = cudaSuccess;
    
    //code
    err = cudaEventCreate(&start);
    if(err != cudaSuccess)
    {
        printf("GPU Error cudaEventCreate() failed - %s.\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaEventCreate(&stop);
    if(err != cudaSuccess)
    {
        printf("GPU Error cudaEventCreate() failed - %s.\n", cudaGetErrorString(err));
        cudaEventDestroy(start);
        exit(EXIT_FAILURE);       
    }

    err = cudaEventRecord(start, 0);
    if(err != cudaSuccess)
    {
        printf("GPU Error cudaEventRecord() failed - %s.\n", cudaGetErrorString(err));
        cleanup();
        exit(EXIT_FAILURE); 
    }

    //allocate device memory
    err = cudaMalloc((void **)&device, bitmap.image_size());
    if(err != cudaSuccess)
    {
        printf("GPU Memory Fatal Error - %s.\n", cudaGetErrorString(err));
        cleanup();
        exit(EXIT_FAILURE);
    }

    //allocate temp memory, initialize it, copy to constant 
    //memory on the GPU, then free our temp memory
    temp = (SPHERE *)malloc(sizeof(SPHERE) * MAX);
    for(int i = 0; i < MAX; i++)
    {
        temp[i].r = rnd(1.0f);
        temp[i].g = rnd(1.0f);
        temp[i].b = rnd(1.0f);
        temp[i].x = rnd(1000.0f) - 500;
        temp[i].y = rnd(1000.0f) - 500;
        temp[i].z = rnd(1000.0f) - 500;
        temp[i].radius = rnd(100.0f) + 20.0f;
    }

    //copy to constant memory 
    err = cudaMemcpyToSymbol(spheres, temp, sizeof(SPHERE) * MAX);

    //kernel configuration
    dim3 DimGrid(DIM / 16, DIM / 16);
    dim3 DimBlock(16, 16);

    kernel<<<DimGrid, DimBlock>>>(device);

    err = cudaMemcpy(bitmap.get_ptr(), device, bitmap.image_size(), cudaMemcpyDeviceToHost);
    if(err != cudaSuccess)
    {
        printf("GPU Memory Fatal Error - %s.\n", cudaGetErrorString(err));
        cleanup();
        exit(EXIT_FAILURE);
    }

    err = cudaEventRecord(stop, 0);
    if(err != cudaSuccess)
    {
        printf("GPU Error cudaEventRecord() failed - %s.\n", cudaGetErrorString(err));
        cleanup();
        exit(EXIT_FAILURE); 
    }

    err = cudaEventSynchronize(stop);
    if(err != cudaSuccess)
    {
        printf("GPU Error cudaEventSynchronize() failed - %s.\n", cudaGetErrorString(err));
        cleanup();
        exit(EXIT_FAILURE); 
    }

    err = cudaEventElapsedTime(&elapsedTime, start, stop);
    if(err != cudaSuccess)
    {
        printf("GPU Error cudaEventElapsedTime() failed - %s.\n", cudaGetErrorString(err));
        cleanup();
        exit(EXIT_FAILURE); 
    }

    printf("Time to generate : %3.1f ms\n", elapsedTime);

    //total cleanup
    cleanup();

    bitmap.display_and_exit();
}

void cleanup(void)
{
    //code
    if(temp)
    {
        free(temp);
        temp = NULL;
    }

    if(device)
    {
        cudaFree(device);
        device = NULL;
    }

    cudaEventDestroy(stop);
    cudaEventDestroy(start);
}
