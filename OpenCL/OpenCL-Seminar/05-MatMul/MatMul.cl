 __kernel void matrixMultiply(__global float *A, __global float *B, __global float *C, int numARows, int numAColumns, int numBRows, int numBColumns, int numCRows, int numCColumns)
{
    //variable declaration
    int row = get_global_id(0);
    int col = get_global_id(1);

    //code
    if((row < numARows) && (col < numBColumns))
    {
        
        float Cvalue = 0.0f;
        for(int k = 0; k < numAColumns; k++)
        {
            Cvalue += A[row * numAColumns + k] * B[k * numBColumns + col];
        }
        C[row * numCColumns + col] = Cvalue;
        
    }
}
