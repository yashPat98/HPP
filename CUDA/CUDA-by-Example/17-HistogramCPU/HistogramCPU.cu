// --- Headers ---
#include "../common/book.h"

#define SIZE (100 * 1024 * 1024)

// --- main() ---
int main(void)
{
    //variable declaration
    unsigned char *buffer = NULL;
    unsigned int histo[256];
    long histoCount = 0;
    clock_t start, stop;
    float elapsedTime;

    //code 
    buffer = (unsigned char *)big_random_block(SIZE);

    //capture the start time
    start = clock();

    for(int i = 0; i < 256; i++)
        histo[i] = 0;

    for(int i = 0; i < SIZE; i++)
        histo[buffer[i]]++;

    //stop timer
    stop = clock();

    elapsedTime = (float)(stop - start) / (float)CLOCKS_PER_SEC * 1000.0f;
    printf("Time to generate : %3.1f ms\n", elapsedTime);

    for(int i = 0; i < 256; i++)
        histoCount += histo[i];

    printf("Histogram Sum : %ld\n", histoCount);

    //free memory 
    free(buffer);
    buffer = NULL;

    return (0);
}
