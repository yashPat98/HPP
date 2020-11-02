#include "../common/book.h"

int main(void)
{
    //variable declaration
    cudaDeviceProp prop;
    int dev;

    //code
    HANDLE_ERROR(cudaGetDevice(&dev));
    printf("ID of current CUDA device : %d\n", dev);

    //zero out structure memory
    memset(&prop, 0, sizeof(cudaDeviceProp));
    prop.major = 1;
    prop.minor = 3;

    HANDLE_ERROR(cudaChooseDevice(&dev, &prop));
    printf("ID of CUDA device closest to revision 1.3 : %d\n", dev);

    HANDLE_ERROR(cudaSetDevice(dev));

    return (0);
}