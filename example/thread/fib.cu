#include <stdio.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include "task_api_all.cuh"

// Enable profiling in the runtime and examples
// #define PROFILE

struct Task {
    int n;
    int result;
};

TASK_DEF(fib, Task) {
    Task c1, c2;
    int a, b;
    int kind1, kind2;
    TASK_BEGIN();
    if (self->n < 2) {
        self->result = self->n;
        TASK_FINISH();
    }
    c1.n = self->n - 1; c1.result = 0;
    c2.n = self->n - 2; c2.result = 0;
    kind1 = 0;
    kind2 = 0;
    // kind1 = c1.n >= 2 ? 0 : 1;
    // kind2 = c2.n >= 2 ? 0 : 1;
    TASK_SPAWN(c1, fib, kind1);
    TASK_SPAWN(c2, fib, kind2);
    TASK_JOIN(0);
    a = TASK_CHILD_RESULT(0);
    b = TASK_CHILD_RESULT(1);
    self->result = a + b;
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

    int n = 40;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    enqueue_initial_task<<<1, 1>>>(n);    // ここはTaskを渡すか？
    execute_task_loop<TERMINATE_ON_FIRST_TASK_FINISH, Task><<<NUM_BLOCKS, THREADS_PER_BLK>>>();
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    cudaDeviceSynchronize();

    int result = 0;
    char* d_tasks_ptr = nullptr;
    size_t task_size = 0;
    cudaMemcpyFromSymbol(&d_tasks_ptr, d_tasks, sizeof(d_tasks_ptr));
    cudaMemcpyFromSymbol(&task_size, d_task_size, sizeof(task_size));
    Task* task_data_ptr = reinterpret_cast<Task*>(d_tasks_ptr + 0 * task_size);
    cudaMemcpy(&result, &task_data_ptr->result, sizeof(int), cudaMemcpyDeviceToHost);
    printf("Fibonacci(%d) = %d\n", n, result);
    printf("Kernel execution time: %.3f ms (%.6f s)\n", milliseconds, milliseconds/1000.0);

    #ifdef PROFILE
    visualize_profile("fib");
    #endif

    // 必要ならdestroy_task_runtimeを呼び出す
    // destroy_task_runtime();
    return 0;
}
