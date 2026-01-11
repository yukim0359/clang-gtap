#include <stdio.h>
#include <cuda_runtime.h>
#define MAX_TASK_SIZE 16
#include "task_api_all.cuh"

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
    for (int i = 0; i < 2; i++) {
        int hoge = 0;
        while (hoge < 4) {
            #pragma gtap task
            add();
            #pragma gtap taskwait
            {
                #pragma gtap task
                add();
                #pragma gtap taskwait
            }
            if (hoge < 3) {
                #pragma gtap task
                add();
                #pragma gtap taskwait
            } else {
                #pragma gtap task
                sub();
                #pragma gtap taskwait
            }
            hoge++;
        }
    }
}

__global__ void my_kernel() {
    #pragma gtap entry
    for_task();
}

int main() {
    // #pragma gtap init
    cudaError_t err = __gtap_init_task_runtime();
    if (err != cudaSuccess) {
        printf("Error: __gtap_init_task_runtime failed\n");
        return 1;
    }

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    my_kernel<<<NUM_BLOCKS, THREADS_PER_BLK>>>();
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
