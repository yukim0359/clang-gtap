#include <stdio.h>
#include <cuda_runtime.h>
#define MAX_TASK_SIZE 16
#include "gtap_thread.cuh"

__device__ int d_result;

#pragma gtap function worker_size(thread)
__device__ void add() {
    atomicAdd(&d_result, 1);
}

#pragma gtap function worker_size(thread)
__device__ void sub() {
    atomicAdd(&d_result, -1);
}

#pragma gtap function worker_size(thread)
__device__ void for_task() {
    d_result = 0;
    {
        // int j = i;
        int j = 0;
        int x = 0;
        #pragma gtap task
        add();
        #pragma gtap taskwait
        if (j < 1) {
            #pragma gtap task
            add();
            #pragma gtap taskwait
        }
    }
}

__global__ void my_kernel() {
    #pragma gtap entry
    for_task();
}

int main() {
    // #pragma gtap init
    cudaError_t err = gtap_initialize();
    if (err != cudaSuccess) {
        printf("Error: __gtap_init_task_runtime failed\n");
        return 1;
    }

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    my_kernel<<<GTAP_GRID_SIZE, GTAP_BLOCK_SIZE>>>();
    cudaEventRecord(stop);
    cudaDeviceSynchronize();
    cudaEventSynchronize(stop);
    
    int h_result;
    cudaMemcpyFromSymbol(&h_result, d_result, sizeof(int));
    printf("Result is %d\n", h_result);

    float elapsed_time;
    cudaEventElapsedTime(&elapsed_time, start, stop);
    printf("Execution time: %f ms\n", elapsed_time);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return 0;
}
