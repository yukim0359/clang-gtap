#include <stdio.h>
#include <cuda_runtime.h>
#include <fstream>
#include <string>

// #define PROFILE

#ifndef SCALING_EVALUATION
#define NUM_BLOCKS 4000
#define THREADS_PER_BLK 32
#define MAX_TASKS_PER_BLOCK 40000
#define MAX_CHILD_TASKS 20300
#define UNNECESSARY_JOIN
// #define MAX_PROFILE_DATA 40000
#endif

#include "task_api_all.cuh"

struct Task {
    int v;
};

// Device-side graph state
__device__ int* g_row_offsets;   // size: num_vertices + 1
__device__ int* g_col_indices;   // size: num_edges
__device__ int* g_depth;         // size: num_vertices; INF indicates unvisited
__device__ int  g_num_vertices;  // number of vertices

TASK_DEF(bfs, Task) {
    TASK_BEGIN();
    const int v = self->v;

    int dv = g_depth[v];
    // printf("bfs: %d (depth: %d)\n", self->v, dv);
    int row_start = g_row_offsets[v];
    int row_end   = g_row_offsets[v + 1];

    for (int e = row_start + threadIdx.x; e < row_end; e += blockDim.x) {
        int u = g_col_indices[e];
        int old = atomicMin(&g_depth[u], dv + 1);
        if (old > dv + 1) {
            Task child{u};
            TASK_SPAWN(child, bfs);
        }
    }

    TASK_FINISH();
    TASK_END();
}

__global__ void enqueue_initial_task(int source) {
    // Root depth = 0 (assume host initialized INF elsewhere)
    g_depth[source] = 0;
    Task initial{source};
    TASK_ENQUEUE_INITIAL(Task, bfs, initial);
}

// Simple demo host: build a chain graph 0->1->2->...->(N-1)
static void build_chain_graph(int N, int** h_row, int** h_col, int* M_out) {
    int* row = (int*)malloc(sizeof(int) * (N + 1));
    int* col = (int*)malloc(sizeof(int) * (N - 1));
    for (int i = 0; i <= N; ++i) row[i] = i < N ? i : (N - 1);
    for (int i = 0; i < N - 1; ++i) col[i] = i + 1;
    *h_row = row;
    *h_col = col;
    *M_out = N - 1;
}

static bool read_csr_binary(const char* path, int** h_row, int** h_col, int* N_out, int* M_out) {
    std::ifstream fin(path, std::ios::binary);
    if (!fin.is_open()) return false;
    int N = 0, M = 0;
    fin.read(reinterpret_cast<char*>(&N), sizeof(int));
    fin.read(reinterpret_cast<char*>(&M), sizeof(int));
    if (!fin) return false;
    int* row = (int*)malloc(sizeof(int) * (N + 1));
    int* col = (int*)malloc(sizeof(int) * M);
    if (row == nullptr || col == nullptr) return false;
    fin.read(reinterpret_cast<char*>(row), sizeof(int) * (N + 1));
    fin.read(reinterpret_cast<char*>(col), sizeof(int) * M);
    if (!fin) { free(row); free(col); return false; }
    *h_row = row; *h_col = col; *N_out = N; *M_out = M;
    return true;
}

int main(int argc, char** argv) {
    std::string csr_path;
    int N = 1000; // default chain graph vertices
    int source = 0;
    if (argc >= 2) csr_path = argv[1];
    if (argc >= 3) source = atoi(argv[2]);

    cudaSetDevice(0);
    cudaError_t st_init = init_task_runtime<Task>();
    if (st_init != cudaSuccess) {
        fprintf(stderr, "init_task_runtime failed: %s\n", cudaGetErrorString(st_init));
        return 1;
    }

    int *h_row = nullptr, *h_col = nullptr; int M = 0;
    if (!csr_path.empty()) {
        if (!read_csr_binary(csr_path.c_str(), &h_row, &h_col, &N, &M)) {
            fprintf(stderr, "Failed to read CSR file: %s\n", csr_path.c_str());
            return 1;
        }
        fprintf(stdout, "Loaded CSR: N=%d, M=%d from %s\n", N, M, csr_path.c_str());
        fflush(stdout);
    } else {
        build_chain_graph(N, &h_row, &h_col, &M);
        fprintf(stdout, "Using synthetic chain graph: N=%d, M=%d\n", N, M);
    }

    int *d_row = nullptr, *d_col = nullptr, *d_depth = nullptr;
    cudaMalloc(&d_row, sizeof(int) * (N + 1));
    cudaMalloc(&d_col, sizeof(int) * M);
    cudaMalloc(&d_depth, sizeof(int) * N);
    cudaMemcpy(d_row, h_row, sizeof(int) * (N + 1), cudaMemcpyHostToDevice);
    cudaMemcpy(d_col, h_col, sizeof(int) * M, cudaMemcpyHostToDevice);

    cudaMemset(d_depth, 0x3f, sizeof(int) * N);

    // Bind device pointers to device symbols
    cudaMemcpyToSymbol(g_row_offsets, &d_row, sizeof(d_row));
    cudaMemcpyToSymbol(g_col_indices, &d_col, sizeof(d_col));
    cudaMemcpyToSymbol(g_depth, &d_depth, sizeof(d_depth));
    cudaMemcpyToSymbol(g_num_vertices, &N, sizeof(int));

    // Timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    enqueue_initial_task<<<1, 1>>>(source);
    execute_task_loop<TERMINATE_ON_ALL_TASKS_FINISH, Task><<<NUM_BLOCKS, THREADS_PER_BLK>>>();

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms = 0.0f; cudaEventElapsedTime(&ms, start, stop);

    int* h_depth = (int*)malloc(sizeof(int) * N);
    cudaMemcpy(h_depth, d_depth, sizeof(int) * N, cudaMemcpyDeviceToHost);
    printf("BFS done. N=%d, source=%d, time=%.3f ms\n", N, source, ms);
    printf("depth[source]=%d\n", h_depth[source]);
    printf("depths from 0 to 10:\n");
    for (int i = 0; i <= 10; i++) {
        printf("depth[%d]=%d\n", i, h_depth[i]);
    }

    #ifdef PROFILE
    visualize_profile("bfs");
    #endif

    free(h_row); free(h_col); free(h_depth);
    cudaFree(d_row); cudaFree(d_col); cudaFree(d_depth);
    return 0;
}


