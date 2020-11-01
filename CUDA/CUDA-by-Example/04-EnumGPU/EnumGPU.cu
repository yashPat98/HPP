#include "../common/book.h"

int main(void)
{
    //variable declaration
    cudaDeviceProp prop;
    int count, i;

    //code
    HANDLE_ERROR(cudaGetDeviceCount(&count));

    for(i = 0; i < count; i++)
    {
        HANDLE_ERROR(cudaGetDeviceProperties(&prop, i));

        printf("--- General Information for device %d ---\n", i);
        printf("Name                  : %s\n", prop.name);
        printf("Compute capability    : %d.%d\n", prop.major, prop.minor);
        printf("Clock rate            : %d\n", prop.clockRate);
        printf("Device copy overlap   : ");
        if(prop.deviceOverlap)
            printf("Enabled\n");
        else
            printf("Disabled\n");
        printf("\n");

        printf("--- Memory Information for device %d ---\n", i);
        printf("Total global mem      : %zd\n", prop.totalGlobalMem);
        printf("Total constant mem    : %zd\n", prop.totalConstMem);
        printf("Max mem pitch         : %zd\n", prop.memPitch);
        printf("Texture Alignment     : %zd\n", prop.textureAlignment);
        printf("\n");
        
        printf("--- MP Information for device %d ---\n", i);
        printf("Multiprocessor count  : %d\n", prop.multiProcessorCount);
        printf("Shared mem per mp     : %zd\n", prop.sharedMemPerBlock);
        printf("Registers per mp      : %d\n", prop.regsPerBlock);
        printf("Threads in warp       : %d\n", prop.warpSize);
        printf("Max threads per block : ", prop.maxThreadsPerBlock);
        printf("Max thread dimensions : (%d, %d, %d)\n", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
        printf("Max grid dimensions   : (%d, %d, %d)\n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
        printf("\n");

    }

    return (0);
}