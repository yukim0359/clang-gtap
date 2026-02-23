#include <stdio.h>
#include <cuda_runtime.h>
#define MAX_TASK_SIZE 16
#include "task_api_all.cuh"

__device__ int d_result;

#pragma gtap function worker_size(thread)
__device__ int fib(int n) {
    if (n < 2) {
        return n;
    }
    int a, b;
    #pragma gtap task
    a = fib(n - 1);
    #pragma gtap task
    b = fib(n - 2);
    #pragma gtap taskwait
    return a + b;
}

__global__ void my_kernel(int n) {
    #pragma gtap entry
    d_result = fib(n);
}

int main() {
    int n = 40;

    // #pragma gtap init
    cudaError_t err = __gtap_init_task_runtime();
    if (err != cudaSuccess) {
        printf("Error: %d\n", err);
        return 1;
    }

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    my_kernel<<<NUM_BLOCKS, THREADS_PER_BLK>>>(n);
    cudaEventRecord(stop);
    cudaDeviceSynchronize();
    cudaEventSynchronize(stop);
    
    int h_result;
    cudaMemcpyFromSymbol(&h_result, d_result, sizeof(int));
    printf("Fibonacci of %d is %d\n", n, h_result);

    float elapsed_time;
    cudaEventElapsedTime(&elapsed_time, start, stop);
    printf("Execution time: %f ms\n", elapsed_time);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return 0;
}
