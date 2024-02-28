#include <iostream>
#include <stdio.h>
#include <stdint.h>
#include <assert.h>
#include <random>
#include <chrono>

// AVX headers
#include <immintrin.h>
#include <x86intrin.h>

#include "iec_units.h"

using namespace std::chrono;


// Choose datatype
//----------------
// #define USE_FLOAT
// #define USE_DOUBLE
#define USE_INT32
// #define USE_INT64

// Choose array size
//------------------
// #define ARRAY_SIZE  _8MiB


#ifdef USE_FLOAT
    typedef float T;
    #define T_print(val) printf("%f ", (val))
#endif
#ifdef USE_DOUBLE
    typedef double T;
    #define T_print(val) printf("%f ", (val))
#endif
#ifdef USE_INT32
    typedef int32_t T;
    #define T_print(val) printf("%3d ", (val))
#endif
#ifndef USE_FLOAT
    #ifndef USE_DOUBLE
        #ifndef USE_INT32
            typedef int64_t T;
            #define T_print(val) printf("%3ld ", (val))
            #define USE_INT64
        #endif
    #endif
#endif


// #define NxN         (ARRAY_SIZE / sizeof(T))
// #define N           ((size_t) floor(sqrt(NxN)))
// #define ACTUAL_SIZE (N*N * sizeof(T))

#define N           1024
#define NxN         (N*N)
#define ARRAY_SIZE  (NxN * sizeof(T))
#define ACTUAL_SIZE ARRAY_SIZE


#define STR_BUFF_OFFSET     16
#define TIME_STR_WIDTH      12




__global__ void gpu_matmul(T* A, T* B, T* C);
__global__ void gpu_matmul_trans(T* A, T* B, T* C);

inline void cpu_matmul(T* A, T* B, T* C);
inline void cpu_matmul_trans(T* A, T* B, T* C);
inline void avx_matmul(T mat1[N][N], T mat2[N][N], T result[N][N]);

void allocate(T** ptr, size_t n);
void print(T* A, T* B, T* C);




int main() {

    // assert(ARRAY_SIZE >= ACTUAL_SIZE);

    // Set locale for printf
    setlocale(LC_NUMERIC, "");
    // String buffer
    char  str_buff[64];
    char* __str_buff = str_buff + STR_BUFF_OFFSET;
    for (size_t i=0; i<STR_BUFF_OFFSET; i++) {
        str_buff[i] = ' ';
    }

    // Query GPU device properties
    int deviceId;
    cudaGetDevice(&deviceId);
    // cudaDeviceProp props;
    // cudaGetDeviceProperties(&props, deviceId);

    // int computeCapabilityMajor = props.major;
    // int computeCapabilityMinor = props.minor;
    // int multiProcessorCount = props.multiProcessorCount;
    // int warpSize = props.warpSize;

    // printf("Device ID: %d\nNumber of SMs: %d\nCompute Capability Major: %d\nCompute Capability Minor: %d\nWarp Size: %d\n\n\n",
    //         deviceId, multiProcessorCount, computeCapabilityMajor, computeCapabilityMinor, warpSize);
    // printf("Number of SMs: %d\nWarp size: %d\n\n", multiProcessorCount, warpSize);


    // Allocate memory
    T* A;
    T* B;
    T* C;
    T* BASELINE;
    allocate(&A, ARRAY_SIZE);
    allocate(&B, ARRAY_SIZE);
    allocate(&C, ARRAY_SIZE);
    allocate(&BASELINE, ARRAY_SIZE);

    #ifdef USE_FLOAT
        printf("Using single-precision floating point");
    #endif
    #ifdef USE_DOUBLE
        printf("Using double-precision floating point");
    #endif
    #ifdef USE_INT32
        printf("Using 32-bit integer");
    #endif
    #ifdef USE_INT64
        printf("Using 64-bit integer");
    #endif
    format_iec(str_buff, ACTUAL_SIZE);
    printf(" with array size %s\n", str_buff);


    // Initialize
    // printf("Initializing...");
    fflush(stdout);
    cudaMemPrefetchAsync(A, ARRAY_SIZE, cudaCpuDeviceId);
    cudaMemPrefetchAsync(B, ARRAY_SIZE, cudaCpuDeviceId);
    cudaMemPrefetchAsync(C, ARRAY_SIZE, cudaCpuDeviceId);
    cudaMemPrefetchAsync(BASELINE, ARRAY_SIZE, cudaCpuDeviceId);
    srand(0);
    #ifdef USE_FLOAT
        std::uniform_real_distribution<T> my_rand(-1,1);
        std::default_random_engine rand_engine;
    #endif
    for (size_t i=0; i<NxN; i++) {
        #ifdef USE_FLOAT
            A[i] = my_rand(rand_engine);
            B[i] = my_rand(rand_engine);
        #else
            if (N <= 8) {
                // Easier to read/debug
                A[i] = rand() % 10;
                B[i] = rand() % 10;
            } else {
                A[i] = rand();
                B[i] = rand();
            }
        #endif
        C[i] = -1;
        BASELINE[i] = -1;
    }
    // printf("done\n");

    // CPU
    printf("CPU...");
    fflush(stdout);
    auto start = high_resolution_clock::now();
    cpu_matmul(A, B, BASELINE);
    auto stop = high_resolution_clock::now();
    auto time = duration_cast<microseconds>(stop - start);
    sprintf(__str_buff, "%'ld", time.count());
    char* time_str = __str_buff - (TIME_STR_WIDTH - strlen(__str_buff));
    printf("done  %s us\n", time_str);

    // CPU Transposed
    printf("CPUt..");
    fflush(stdout);
    start = high_resolution_clock::now();
    cpu_matmul_trans(A, B, C);
    stop = high_resolution_clock::now();
    time = duration_cast<microseconds>(stop - start);
    sprintf(__str_buff, "%'ld", time.count());
    time_str = __str_buff - (TIME_STR_WIDTH - strlen(__str_buff));
    printf("done  %s us\n", time_str);


    // GPU
    printf("GPU...");
    fflush(stdout);
    cudaMemPrefetchAsync(A, ARRAY_SIZE, deviceId);
    cudaMemPrefetchAsync(B, ARRAY_SIZE, deviceId);
    cudaMemPrefetchAsync(C, ARRAY_SIZE, deviceId);
    start = high_resolution_clock::now();
    gpu_matmul<<<dim3(N,N),1>>>(A, B, C);
    cudaDeviceSynchronize();
    stop = high_resolution_clock::now();
    time = duration_cast<microseconds>(stop - start);
    sprintf(__str_buff, "%'ld", time.count());
    time_str = __str_buff - (TIME_STR_WIDTH - strlen(__str_buff));
    printf("done  %s us\n", time_str);


    // GPU Transposed
    printf("GPUt..");
    fflush(stdout);
    cudaMemPrefetchAsync(A, ARRAY_SIZE, deviceId);
    cudaMemPrefetchAsync(B, ARRAY_SIZE, deviceId);
    cudaMemPrefetchAsync(C, ARRAY_SIZE, deviceId);
    start = high_resolution_clock::now();
    gpu_matmul_trans<<<dim3(N,N),1>>>(A, B, C);
    cudaDeviceSynchronize();
    stop = high_resolution_clock::now();
    time = duration_cast<microseconds>(stop - start);
    sprintf(__str_buff, "%'ld", time.count());
    time_str = __str_buff - (TIME_STR_WIDTH - strlen(__str_buff));
    printf("done  %s us\n", time_str);


    // AVX
    #ifdef USE_INT32
        cudaMemPrefetchAsync(A, ARRAY_SIZE, cudaCpuDeviceId);
        cudaMemPrefetchAsync(B, ARRAY_SIZE, cudaCpuDeviceId);
        cudaMemPrefetchAsync(C, ARRAY_SIZE, cudaCpuDeviceId);
        printf("AVX...");
        fflush(stdout);
        // Requires C to be initialized to 0
        for (size_t i=0; i<NxN; i++) {
            C[i] = 0;
        }
        start = high_resolution_clock::now();
        avx_matmul((T(*)[N]) A, (T(*)[N]) B, (T(*)[N]) C);
        stop = high_resolution_clock::now();
        time = duration_cast<microseconds>(stop - start);
        sprintf(__str_buff, "%'ld", time.count());
        time_str = __str_buff - (TIME_STR_WIDTH - strlen(__str_buff));
        printf("done  %s us\n", time_str);
        
        // Verify
        printf("Verifying AVX...");
        fflush(stdout);
        // cudaMemPrefetchAsync(A, ARRAY_SIZE, cudaCpuDeviceId);
        // cudaMemPrefetchAsync(B, ARRAY_SIZE, cudaCpuDeviceId);
        // cudaMemPrefetchAsync(C, ARRAY_SIZE, cudaCpuDeviceId);
        bool valid = true;
        for (size_t row=0; row<N; row++) {
            for (size_t col=0; col<N; col++) {
                if (C[row*N + col] != BASELINE[row*N + col]) {
                    valid = false;
                }
            }
        }
        if (!valid) {
            printf("\nERROR: CPU and AVX results do not match.\n");
            if (N <= 16) {
                printf("\nCPU:\n");
                print(A, B, BASELINE);
            }
            if (N <= 16) {
                printf("\n\nAVX:\n");
                print(A, B, C);
            }
        } else {
            printf("done\n");
        }
    #endif

}


__global__ void gpu_matmul(T* A, T* B, T* C) {
    T sum = 0;
    for (size_t k=0; k<N; k++) {
        sum += A[blockIdx.x*N + k] * B[k*N + blockIdx.y]; // B not transposed
    }
    C[blockIdx.x*N + blockIdx.y] = sum;
}


__global__ void gpu_matmul_trans(T* A, T* B, T* C) {
    T sum = 0;
    for (size_t k=0; k<N; k++) {
        sum += A[blockIdx.x*N + k] * B[k + N*blockIdx.y]; // B tranposed
    }
    C[blockIdx.x*N + blockIdx.y] = sum;
}


inline void cpu_matmul(T* A, T* B, T* C) {
    for (size_t row=0; row<N; row++) {
        for (size_t col=0; col<N; col++) {
            C[row*N+col] = 0.0;
            for (size_t k=0; k<N; k++) {
                C[row*N+col] += A[row*N+k] * B[k*N+col];
            }
        }
    }
}


inline void cpu_matmul_trans(T* A, T* B, T* C) {
    for (size_t row=0; row<N; row++) {
        for (size_t col=0; col<N; col++) {
            C[row*N+col] = 0.0;
            for (size_t k=0; k<N; k++) {
                C[row*N+col] += A[row*N+k] * B[k+N*col];
            }
        }
    }
}


inline void avx_matmul(T mat1[N][N], T mat2[N][N], T result[N][N]) {
    __m256i vec_multi_res = _mm256_setzero_si256(); //Initialize vector to zero
    __m256i vec_mat1 = _mm256_setzero_si256(); //Initialize vector to zero
    __m256i vec_mat2 = _mm256_setzero_si256(); //Initialize vector to zero

    size_t i, j, k;
    for (i = 0; i < N; i++)
    {
        for (j = 0; j < N; ++j)
        {
            //Stores one element in mat1 and use it in all computations needed before proceeding
            //Stores as vector to increase computations per cycle
            vec_mat1 = _mm256_set1_epi32(mat1[i][j]);

            for (k = 0; k < N; k += 8)
            {
                vec_mat2 = _mm256_loadu_si256((__m256i*)&mat2[j][k]); //Stores row of second matrix (eight in each iteration)
                vec_multi_res = _mm256_loadu_si256((__m256i*)&result[i][k]); //Loads the result matrix row as a vector
                vec_multi_res = _mm256_add_epi32(vec_multi_res ,_mm256_mullo_epi32(vec_mat1, vec_mat2));//Multiplies the vectors and adds to the result vector

                _mm256_storeu_si256((__m256i*)&result[i][k], vec_multi_res); //Stores the result vector into the result array
            }
        }
    }
}


inline void avx_matmul_intel(T* A, T* B, T* C) {

}


void allocate(T** ptr, size_t size) {
    cudaError_t err = cudaMallocManaged(ptr, size);
    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
        assert(NULL != *ptr);
    }
}


void print(T* A, T* B, T* C) {
    for (size_t row=0; row<N; row++) {
        printf("| ");
        for (size_t col=0; col<N; col++) {
            // printf("%d ", A[row*N+col]);
            T_print(A[row*N+col]);
        }
        printf("|  | ");
        for (size_t col=0; col<N; col++) {
            // printf("%d ", B[row*N+col]);
            T_print(B[row*N+col]);
        }
        printf("|  =  | ");
        for (size_t col=0; col<N; col++) {
            // printf("%d ", C[row*N+col]);
            T_print(C[row*N+col]);
        }
        printf("|\n");
    }
    printf("\n");
}
