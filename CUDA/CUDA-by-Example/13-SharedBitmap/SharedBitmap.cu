//headers
#include <cuda.h>
#include "../common/book.h"
#include "../common/cpu_bitmap.h"

#define DIM 1024
#define PI  3.14159265

// *** CUDA KERNEL DEFINITION ***
__global__ void kernel(unsigned char *ptr)
{
    //variable declaration
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.x;
    int offset = x + y * blockDim.x * gridDim.x;

    __shared__ float shared[16][16];

    //code
    const float period = 128.0f;

    shared[threadIdx.x][threadIdx.y] = 255 * (sinf(x * 2.0f * PI / period) + 1.0f) * 
                                             (sinf(y * 2.0f * PI / period) + 1.0f) / 4.0f;

    __syncthreads();

    ptr[offset * 4 + 0] = 0;
    ptr[offset * 4 + 1] = shared[15 - threadIdx.x][15 - threadIdx.y];
    ptr[offset * 4 + 2] = 0;
    ptr[offset * 4 + 3] = 255;
}

int main(void)
{
    //variable declaration
    CPUBitmap bitmap(DIM, DIM);
    unsigned char *dev_bitmap;
    cudaError_t err = cudaSuccess;

    //code
    //allocate device memory
    err = cudaMalloc((void **)&dev_bitmap, bitmap.image_size());
    if(err != cudaSuccess)
    {
        printf("GPU Memory Fatal Error - %s In The File Name %s At Line No %d.\n", cudaGetErrorString(err), __FILE__, __LINE__);
        exit(EXIT_FAILURE);
    }

    //kernel configuration
    dim3 DimGrid(DIM / 16, DIM / 16);
    dim3 DimBlock(16, 16);

    kernel<<<DimGrid, DimBlock>>>(dev_bitmap);

    err = cudaMemcpy(bitmap.get_ptr(), dev_bitmap, bitmap.image_size(), cudaMemcpyDeviceToHost);
    if(err != cudaSuccess)
    {
        printf("GPU Memory Fatal Error - %s In The File Name %s At Line No %d.\n", cudaGetErrorString(err), __FILE__, __LINE__);
        cudaFree(dev_bitmap);
        dev_bitmap = NULL;
        exit(EXIT_FAILURE);
    }

    cudaFree(dev_bitmap);
    dev_bitmap = NULL;

    bitmap.display_and_exit();
}
