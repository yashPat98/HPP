#include <iostream>
#include "../common/book.h"

__global__ void add(int a, int b, int *c)
{
    //code
    *c = a + b;
}

int main(void)
{
    //variable declaration
    int c;
    int *dev_c = NULL;

    //code 
    HANDLE_ERROR(cudaMalloc((void**)&dev_c, sizeof(int)));

    add<<<1,1>>>(2, 7, dev_c);

    HANDLE_ERROR(cudaMemcpy(&c, dev_c, sizeof(int), cudaMemcpyDeviceToHost));

    printf("2 + 7 = %d\n", c);

    cudaFree(dev_c);
    dev_c = NULL;

    return (0);
}
