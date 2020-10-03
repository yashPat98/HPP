//headers
#include <stdio.h>
#include <stdlib.h>
#include <CL/opencl.h>          //standard OpenCL header file

//global variable declaration
cl_int           ret_ocl;
cl_int           platform_count;
cl_platform_id  *cl_platform_ids = NULL;
cl_platform_id   oclPlatformID_required;

cl_int           device_count;
cl_device_id    *cl_device_ids = NULL;
cl_device_id     oclComputeDeviceID;
cl_device_type   device_type;

char PlatformInfo[512];
char DeviceInfo[512];

int main(int argc, char *argv[])
{
    //function declaration
    void cleanup(void);

    //code
    ret_ocl = clGetPlatformIDs(0, NULL, &platform_count);
    if(ret_ocl != CL_SUCCESS)
    {
        printf("OpenCL Error - clGetPlatformIDs() failed : %d. Exiting Now ...\n", ret_ocl);
        exit(EXIT_FAILURE);
    }

    //check the platform count
    if(platform_count < 1)
    {
        printf("There Are No OpenCL Supporting Platforms.\n");
        exit(EXIT_FAILURE);
    }
    
    //allocated the memory according to number of platforms
    cl_platform_ids = (cl_platform_id *)malloc(platform_count * sizeof(cl_platform_id));
    if(cl_platform_ids == NULL)
    {
        printf("Memory Allocation For cl_platform_ids Failed. Exiting Now...\n");
        exit(EXIT_FAILURE);
    }

    //calling clGetPlatformIDs() again with proper platform count
    ret_ocl = clGetPlatformIDs(platform_count, cl_platform_ids, NULL);
    if(ret_ocl != CL_SUCCESS)
    {
        printf("OpenCL Error - clGetPlatformIDs() failed : %d. Exiting Now ...\n", ret_ocl);
        cleanup();
        exit(EXIT_FAILURE);
    }

    //ennumerate platforms to get devices
    for(int i = 0; i < platform_count; i++)
    {
        clGetPlatformInfo(cl_platform_ids[i], CL_PLATFORM_NAME, 512, PlatformInfo, NULL);
        
        printf("OpenCL Platform = %s.\n", PlatformInfo);

        //get device count of respective platform
        ret_ocl = clGetDeviceIDs(cl_platform_ids[i], CL_DEVICE_TYPE_ALL, 0, NULL, &device_count);
        if(ret_ocl != CL_SUCCESS)
        {
            printf("OpenCL Error - clGetDeviceIDs() failed : %d. Exiting Now ...\n", ret_ocl);
            cleanup();
            exit(EXIT_FAILURE);
        }

        if(device_count < 1)
        {
            printf("%s Platform Does Not Have OpenCL Compatible Device. Exiting Now ...\n", cl_platform_ids[i]);
            continue;
        }

        //allocate memory according to number of devices
        cl_device_ids = (cl_device_id *)malloc(device_count * sizeof(cl_device_id));
        if(cl_device_ids == NULL)
        {
            printf("Memory Allocation For cl_device_ids failed. Exiting Now ...\n");
            cleanup();
            exit(EXIT_FAILURE);
        }

        for(int j = 0; j < device_count; j++)
        {
            clGetDeviceInfo(cl_device_ids[j], CL_DEVICE_NAME, 512, DeviceInfo, NULL);

            printf("%s Platform Has %s OpenCL Compatible Device.\n", PlatformInfo, DeviceInfo);

            clGetDeviceInfo(cl_device_ids[j], CL_DEVICE_TYPE, sizeof(cl_device_type), &device_type, NULL);

            //decide the type you want 
            if(device_type & CL_DEVICE_TYPE_GPU)
            {
                oclPlatformID_required = cl_platform_ids[i];
                oclComputeDeviceID = cl_device_ids[j];

                break;
            }
        }
    } 

    cleanup();

    return (0);
}

void cleanup(void)
{
    //code
    //free memory for cl_device_ids
    if(cl_device_ids)
    {
        free(cl_device_ids);
        cl_device_ids = NULL;
    }

    //free memory for cl_platform_ids
    if(cl_platform_ids)
    {
        free(cl_platform_ids);
        cl_platform_ids = NULL;
    }
}
