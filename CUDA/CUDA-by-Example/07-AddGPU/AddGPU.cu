//headers
#include "../common/book.h"

#define N 10

int main(void)
{
    //function declaration
    __global__ void add(int *a, int *b, int *c);

    //variable declaration
    int i, a[N], b[N], c[N];
    int *dev_a = NULL;
    int *dev_b = NULL;
    int *dev_c = NULL;

    //allocate the memory on the GPU
    HANDLE_ERROR(cudaMalloc((void**)&dev_a, N * sizeof(int)));
    HANDLE_ERROR(cudaMalloc((void**)&dev_b, N * sizeof(int)));
    HANDLE_ERROR(cudaMalloc((void**)&dev_c, N * sizeof(int)));

    //fill the arrays 'a' and 'b' on the CPU
    for(i = 0; i < N; i++)
    {
        a[i] = -i;
        b[i] = i * i;
    }

    //copy the arrays 'a' and 'b' to the GPU
    HANDLE_ERROR(cudaMemcpy(dev_a, a, N * sizeof(int), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(dev_b, b, N * sizeof(int), cudaMemcpyHostToDevice));

    add<<<N, 1>>>(dev_a, dev_b, dev_c);

    //copy the array 'c' back from the GPU to the CPU
    HANDLE_ERROR(cudaMemcpy(c, dev_c, N * sizeof(int), cudaMemcpyDeviceToHost));

    //display the results
    for(i = 0; i < N; i++)
    {
        printf("%d + %d = %d\n", a[i], b[i], c[i]);
    }

    //free the memory
    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);

    dev_a = NULL;
    dev_b = NULL;
    dev_c = NULL;

    return (0);
}

__global__ void add(int *a, int *b, int *c)
{
    //variable declaration
    int tid = blockIdx.x;               //handle the data at this index

    //code
    if(tid < N)
    {
        c[tid] = a[tid] + b[tid];
    }
}

