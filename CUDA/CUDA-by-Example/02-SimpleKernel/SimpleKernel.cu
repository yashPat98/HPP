#include <iostream>

__global__ void kernel(void)
{
    //code
}

int main(void)
{
    //code
    kernel<<<1,1>>>();
    printf("Hello World\n");
    return (0);
}
