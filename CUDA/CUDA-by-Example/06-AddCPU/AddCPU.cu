//headers
#include <iostream>
using namespace std;

#define N 10

//main()
int main(void)
{
    //function prototypes
    void add(int *a, int *b, int *c);

    //variable declaration
    int i, a[N], b[N], c[N];

    //code
    //fill the arrays a and b on the CPU
    for(i = 0; i < N; i++)
    {
        a[i] = -i;
        b[i] = i * i;
    }

    add(a, b, c);

    //display the result
    for(i = 0; i < N; i++)
    {
        printf("%d + %d = %d\n", a[i], b[i], c[i]);
    }

    return (0);
}

void add(int *a, int *b, int *c)
{
    //variable declaration
    int tid = 0;                    //this is CPU zero, so we start at zero

    //code
    while(tid < N)
    {
        c[tid] = a[tid] + b[tid];
        tid += 1;                   //we have one CPU so we increment by one
    }
}
