#include <stdio.h>
#include <cuda_runtime.h>

struct Task {
    int row;
    int column;
    int left;
    int right;
    int down;
};

#ifdef USE_GQ
#include "task_api_all_gq.cuh"
#else
#include "task_api_all.cuh"
#endif

__device__ int  d_answer;
__device__ int  d_grid_size;
__device__ int  d_cutoff_depth;

__device__ void serial_search(int row, int column, int left, int down, int right) {
    int grid_size = d_grid_size;
    if (row == grid_size) {
        atomicAdd(&d_answer, 1);
        return;
    }
    uint32_t mask = (grid_size < 32 ? (1u << grid_size) - 1 : 0xFFFFFFFFu);
    uint32_t avail = mask & ~((uint32_t)column | (uint32_t)left | (uint32_t)right);
    while (avail) {
        uint32_t p = avail & -avail;
        avail -= p;
        serial_search(row + 1, column | p, (left | p) << 1, down | p, (right | p) >> 1);
    }
}

TASK_DEF(nq, Task) {
    uint32_t mask, avail;
    int grid_size = d_grid_size;
    int cutoff_depth = d_cutoff_depth;
    TASK_BEGIN();
    if (self->row > cutoff_depth) {
        serial_search(self->row, self->column, self->left, self->down, self->right);
        TASK_FINISH();
    }
    if (self->row == grid_size) {
        atomicAdd(&d_answer, 1);
        TASK_FINISH();
    }
    mask = (grid_size < 32 ? (1u << grid_size) - 1 : 0xFFFFFFFFu);
    avail = mask & ~((uint32_t)self->column | (uint32_t)self->left | (uint32_t)self->right);
    // if (avail == 0) {
    //     TASK_FINISH();
    // }
    while (avail) {
        uint32_t p = avail & -avail;
        avail -= p;
        Task child;
        child.row = self->row + 1;
        child.column = self->column | (int)p;
        child.left = (self->left | (int)p) << 1;
        child.down = self->down | (int)p;
        child.right = (self->right | (int)p) >> 1;
        // int kind = 0;
        // int kind = child.row > cutoff_depth ? 1 : 0;
        TASK_SPAWN(child, nq, 0);
    }
    // TASK_JOIN();
    TASK_FINISH();
    TASK_END();
}

__global__ void enqueue_initial_task() {
    Task initialTask = {.row = 0, .column = 0, .left = 0, .right = 0, .down = 0};
    TASK_ENQUEUE_INITIAL(Task, nq, initialTask);
}

int main(int argc, char **argv) {
    // Parse command line arguments
    int GRID_SIZE = (argc > 1 ? atoi(argv[1]) : 16);
    int CUTOFF_DEPTH = 7;
    
    cudaSetDevice(0);
    cudaError_t cudaStatus;

    cudaStatus = init_task_runtime<Task>();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "init_task_runtime failed! %s\n", cudaGetErrorString(cudaStatus));
        return 1;
    }

    int zero = 0;
    cudaStatus = cudaMemcpyToSymbol(d_answer, &zero, sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "Answer initialization failed! %s\n", cudaGetErrorString(cudaStatus));
        return 1;
    }
    
    // Set grid size on device
    cudaStatus = cudaMemcpyToSymbol(d_grid_size, &GRID_SIZE, sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "Grid size initialization failed! %s\n", cudaGetErrorString(cudaStatus));
        return 1;
    }
    
    // Set cutoff depth on device
    cudaStatus = cudaMemcpyToSymbol(d_cutoff_depth, &CUTOFF_DEPTH, sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "Cutoff depth initialization failed! %s\n", cudaGetErrorString(cudaStatus));
        return 1;
    }

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    enqueue_initial_task<<<1, 1>>>();
    execute_task_loop<TERMINATE_ON_ALL_TASKS_FINISH, Task><<<NUM_BLOCKS, THREADS_PER_BLK>>>();

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "Kernel execution failed! %s\n", cudaGetErrorString(cudaStatus));
        return 1;
    }

    int result;
    cudaStatus = cudaMemcpyFromSymbol(&result, d_answer, sizeof(int), 0, cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "Result retrieval failed! %s\n", cudaGetErrorString(cudaStatus));
        return 1;
    }

    printf("N-Queens(%d) = %d\n", GRID_SIZE, result);
    printf("Kernel execution time: %.3f ms (%.6f s)\n", milliseconds, milliseconds/1000.0);

    #ifdef PROFILE
    visualize_profile("nq");
    #endif

    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }
    return 0;
}
