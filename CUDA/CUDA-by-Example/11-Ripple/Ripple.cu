//headers
#include <cuda.h>
#include "../common/book.h"
#include "../common/cpu_anim.h"

#define DIM 1024
#define PI 3.14159265f

//global variables
struct DataBlock
{
    unsigned char *dev_bitmap;
    CPUAnimBitmap *bitmap;
};

//*** CUDA KERNEL DEFINITION ***
__global__ void kernel(unsigned char *ptr, int ticks)
{
    //map from threadIdx / BlockIdx to pixel position
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y; 
    int offset = x + y * blockDim.x * gridDim.x;

    //now calculate the value at that position
    float fx = x - DIM / 2;
    float fy = y - DIM / 2;
    float d = sqrtf(fx * fx + fy * fy);

    unsigned char grey = (unsigned char)(128.0f + 127.0f *
                                        cos(d / 10.0f - ticks / 7.0f) / 
                                        (d / 10.0f + 1.0f));

    ptr[offset * 4 + 0] = grey;
    ptr[offset * 4 + 1] = grey;
    ptr[offset * 4 + 2] = grey;
    ptr[offset * 4 + 3] = 255;

}

int main(void)
{
    //function declaration
    void cleanup(DataBlock *d);
    void generate_frame(DataBlock *d, int ticks);

    //variable declaration
    DataBlock data;
    CPUAnimBitmap bitmap(DIM, DIM, &data);
    data.bitmap = &bitmap;

    //code 
    cudaError_t err = cudaSuccess;
    err = cudaMalloc((void **)&data.dev_bitmap, bitmap.image_size());
    if(err != cudaSuccess)
    {
        printf("GPU Memory Fatal Error = %s In File Name %s At Line No %d.\nExiting Now ...\n", cudaGetErrorString(err), __FILE__, __LINE__);
        exit(EXIT_FAILURE);
    }

    bitmap.anim_and_exit((void (*)(void *, int))generate_frame, (void (*)(void*))cleanup);

    return (0);
}

void generate_frame(DataBlock *d, int ticks)
{
    //function declaration
    void cleanup(DataBlock *d);

    //variable declaration
    cudaError_t err;
    dim3 DimGrid(DIM / 16, DIM / 16);
    dim3 DimBlock(16, 16);

    //code
    //launch the kernel
    kernel<<<DimGrid, DimBlock>>>(d->dev_bitmap, ticks);
    err = cudaMemcpy(d->bitmap->get_ptr(), 
                     d->dev_bitmap,
                     d->bitmap->image_size(),
                     cudaMemcpyDeviceToHost);
    
    if(err != cudaSuccess)
    {
        printf("GPU Memory Fatal Error = %s In File Name %s At Line No %d.\nExiting Now ...\n", cudaGetErrorString(err), __FILE__, __LINE__);
        cleanup(d);
        exit(EXIT_FAILURE);
    }

}

void cleanup(DataBlock *d)
{
    if(d->dev_bitmap)
    {
        cudaFree(d->dev_bitmap);
        d->dev_bitmap = NULL;
    }
}
