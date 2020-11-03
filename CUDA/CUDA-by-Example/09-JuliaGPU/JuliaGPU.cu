//headers 
#include "../common/book.h"
#include "../common/cpu_bitmap.h"

#define DIM 1000

struct cuComplex
{
    float r;
    float i;

    __device__ cuComplex(float a, float b) : r(a), i(b) {}
    
    __device__ float magnitude2(void)
    {
        return (r * r + i * i);
    } 

    __device__ cuComplex operator+(const cuComplex& a)
    {
        return cuComplex(r + a.r, i + a.i);
    }

    __device__ cuComplex operator*(const cuComplex& a)
    {
        return cuComplex(r * a.r - i * a.i, i * a.r + r * a.i);
    }
};

//main
int main(void)
{
    //function prototypes
    __global__ void kernel(unsigned char *ptr);

    //variable declaration
    CPUBitmap bitmap(DIM, DIM);
    unsigned char *dev_ptr = NULL;

    //code
    //allocate memory on device
    HANDLE_ERROR(cudaMalloc((void**)&dev_ptr, bitmap.image_size()));

    dim3 grid(DIM, DIM);
    kernel<<<grid, 1>>>(dev_ptr);

    HANDLE_ERROR(cudaMemcpy(bitmap.get_ptr(), dev_ptr, bitmap.image_size(), cudaMemcpyDeviceToHost));

    bitmap.display_and_exit();

    HANDLE_ERROR(cudaFree(dev_ptr));

    return (0);
}

__global__ void kernel(unsigned char *ptr)
{
    //function prototypes
    __device__ int julia(int x, int y);

    //code
    //map from threadIdx/blockIdx to pixel position
    int x = blockIdx.x;
    int y = blockIdx.y;
    int offset = x + y * gridDim.x;

    //now calculate the value at that position
    int juliaValue = julia(x, y);

    ptr[offset * 4 + 0] = 0;                    //R
    ptr[offset * 4 + 1] = 0;                    //G
    ptr[offset * 4 + 2] = 255 * juliaValue;     //B
    ptr[offset * 4 + 3] = 255;                  //A
}

__device__ int julia(int x, int y)
{
    //code
    const float scale = 1.5;
    float jx = scale * (float)(DIM/2 - x) / (DIM/2);
    float jy = scale * (float)(DIM/2 - y) / (DIM/2);

    cuComplex c(-0.8, 0.156);
    cuComplex a(jx, jy);

    int i = 0;
    for(i = 0; i < 200; i++)
    {
        a = a * a + c;
        if(a.magnitude2() > 1000)
            return (0);
    }

    return (1);
}
