#include <stdio.h>

#define N 2048 * 2048 // Number of elements in each vector

/*
 * Optimize this already-accelerated codebase. Work iteratively,
 * and use nsys to support your work.
 *
 * Aim to profile `saxpy` (without modifying `N`) running under
 * 200,000 ns.
 *
 * Some bugs have been placed in this codebase for your edification.
 */

__global__
void initMem(int* a, int* b, int* c) {
    int index = blockIdx.x*blockDim.x + threadIdx.x;
    int stride = blockDim.x*gridDim.x;

    for (int i=index; i<N; i+=stride) {
        a[i] = 2;
        b[i] = 1;
        c[i] = 0;
    }
}

__global__
void saxpy(int* a, int* b, int* c) {
    int index = blockIdx.x*blockDim.x + threadIdx.x;
    int stride = blockDim.x*gridDim.x;

    for (int i=index; i<N; i+=stride) {
        c[i] = 2*a[i] + b[i];
    }
}

int main()
{
    int deviceId;
    cudaGetDevice(&deviceId);
    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, deviceId);
    int SMs = props.multiProcessorCount;

    int *a, *b, *c;

    int size = N * sizeof (int); // The total number of bytes per vector
    
    cudaMallocManaged(&a, size);
    cudaMallocManaged(&b, size);
    cudaMallocManaged(&c, size);
    
    int threads_per_block = 1024;
    int number_of_blocks = 32 * SMs;

    // Initialize memory
    // for(int i=0; i<N; i++ ) {
    //     a[i] = 2;
    //     b[i] = 1;
    //     c[i] = 0;
    // }

    cudaMemPrefetchAsync(a, size, deviceId);
    cudaMemPrefetchAsync(b, size, deviceId);
    cudaMemPrefetchAsync(c, size, deviceId);

    initMem<<<number_of_blocks, threads_per_block>>>(a, b, c);

    cudaDeviceSynchronize();
    printf("Initialized\n");

    saxpy<<<number_of_blocks, threads_per_block>>>( a, b, c );

    cudaDeviceSynchronize();
    printf("Calculation complete\n");
    cudaMemPrefetchAsync(c, size, cudaCpuDeviceId);

    // Print out the first and last 5 values of c for a quality check
    for( int i = 0; i < 5; ++i )
        printf("c[%d] = %d, ", i, c[i]);
    printf ("\n");
    for( int i = N-5; i < N; ++i )
        printf("c[%d] = %d, ", i, c[i]);
    printf ("\n");

    cudaFree( a ); cudaFree( b ); cudaFree( c );
}
