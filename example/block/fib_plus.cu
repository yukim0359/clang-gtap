#include <stdio.h>
#include <math.h>
#include <cuda_runtime.h>

// #define PROFILE

#define DATA_LENGTH 2048

#ifdef USE_GQ
#include "task_api_all_gq.cuh"
#else
#include "task_api_all.cuh"
#endif

struct Task {
    int   n;
    int   result;
};

__device__ bool heavy_operation() {
    // Use FMA-heavy math on shared data to better exercise the GPU pipelines.
    __shared__ float sdata[DATA_LENGTH];
    float acc = static_cast<float>(threadIdx.x + 1);

    // Write phase: each thread strides through the array doing FMA work.
    for (int i = threadIdx.x; i < DATA_LENGTH; i += blockDim.x) {
        float base = static_cast<float>(i) * 1.001f;
        float val = __fmaf_rn(base, acc, 0.5f); // val = base * acc + 0.5
        sdata[i] = val;
        acc = __fmaf_rn(acc, 1.0001f, 0.123f);
    }

    __syncthreads();

    // Verification phase: recompute expected values and check with tolerance.
    acc = static_cast<float>(threadIdx.x + 1);
    for (int i = threadIdx.x; i < DATA_LENGTH; i += blockDim.x) {
        float base = static_cast<float>(i) * 1.001f;
        float expected = __fmaf_rn(base, acc, 0.5f);
        if (fabsf(sdata[i] - expected) > 1e-3f) {
            printf("FATAL ERROR: heavy_operation failed at index %d\n", i);
            __trap();
        }
        acc = __fmaf_rn(acc, 1.0001f, 0.123f);
    }

    return true;
}

TASK_DEF(fib, Task) {
    Task c1, c2;
    int a, b;
    TASK_BEGIN();
    if (self->n < 2) {
        if (!heavy_operation()) {
            printf("FATAL ERROR: heavy_operation failed\n");
            __trap();
        }
        if(threadIdx.x == 0) self->result = self->n;
        __syncthreads();
        TASK_FINISH();
    }
    if (threadIdx.x == 0) {
        c1.n = self->n - 1; c1.result = 0;
        c2.n = self->n - 2; c2.result = 0;
        TASK_SPAWN(c1, fib);
        TASK_SPAWN(c2, fib);
    }
    __syncthreads();
    TASK_JOIN();

    if (!heavy_operation()) {
        printf("FATAL ERROR: heavy_operation failed\n");
        __trap();
    }
    if (threadIdx.x == 0) {
        a = TASK_CHILD_RESULT(0);
        b = TASK_CHILD_RESULT(1);
        self->result = a + b;
    }
    __syncthreads();
    TASK_FINISH();
    TASK_END();
}

__global__ void enqueue_initial_task(int n) {
    Task initialTask = {.n = n, .result = 0};
    TASK_ENQUEUE_INITIAL(Task, fib, initialTask);
}

int main(int argc, char** argv) {
    cudaSetDevice(0);
    cudaError_t st_init = init_task_runtime<Task>();
    if (st_init != cudaSuccess) {
        fprintf(stderr, "init_task_runtime failed: %s\n", cudaGetErrorString(st_init));
        return 1;
    }

    int n = 30; // sample
    if (argc >= 2) n = atoi(argv[1]);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    enqueue_initial_task<<<1, 1>>>(n);    // ここはTaskを渡すか？
    execute_task_loop<TERMINATE_ON_ALL_TASKS_FINISH, Task><<<NUM_BLOCKS, THREADS_PER_BLK>>>();
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    cudaDeviceSynchronize();

    int result = 0;
    Task* d_task_data_bytes_ptr = nullptr;
    cudaMemcpyFromSymbol(&d_task_data_bytes_ptr, d_task_data_bytes, sizeof(d_task_data_bytes_ptr));
    cudaMemcpy(&result, &d_task_data_bytes_ptr[0].result, sizeof(int), cudaMemcpyDeviceToHost);
    printf("Fibonacci(%d) = %d\n", n, result);
    printf("Kernel execution time: %.3f ms (%.6f s)\n", milliseconds, milliseconds/1000.0);

    #ifdef PROFILE
    visualize_profile("fib_plus");
    #endif

    return 0;
}
