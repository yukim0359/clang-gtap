#include <stdio.h>
#include <cuda_runtime.h>
#include <fstream>
#include <string>
#include <queue>
#include <algorithm>
#include "task_api_all.cuh"

struct Task {
    int v;
};

// Device-side graph state
__device__ int* g_row_offsets;   // size: num_vertices + 1
__device__ int* g_col_indices;   // size: num_edges
__device__ int* g_depth;         // size: num_vertices; INF indicates unvisited
__device__ int  g_num_vertices;  // number of vertices

// Helper to chunk spawns to respect MAX_CHILD_TASKS
TASK_DEF(bfs, Task) {
    int u, old;
    int dv, row_start, row_end;
    int v;
    TASK_BEGIN();

    v = self->v;
    dv = load_L2(&g_depth[v]);
    row_start = g_row_offsets[v];
    row_end   = g_row_offsets[v + 1];

    for (int e = row_start; e < row_end; ++e) {
        u = g_col_indices[e];
        old = atomicMin(&g_depth[u], dv + 1);
        if (old > dv + 1) {
            Task child{u};
            TASK_SPAWN(child, bfs, 0);
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
    #ifdef COUNT_TASKS
    unsigned long long zero_task_num = 0ull;
    cudaMemcpyToSymbol(g_task_num, &zero_task_num, sizeof(unsigned long long));
    #endif

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
    printf("depth[source]=%d, depth[%d]=%d, depth[%d]=%d, depth[%d]=%d, depth[%d]=%d, depth[%d]=%d\n",
           h_depth[source], 1, h_depth[1], 2, h_depth[2], 3, h_depth[3], 4, h_depth[4], 5, h_depth[5]);

    // Validation: Compare with CPU reference implementation
    printf("\n=== BFS Validation ===\n");
    int* h_depth_cpu = (int*)malloc(sizeof(int) * N);
    const int INF_CPU = 0x3f3f3f3f;
    for (int i = 0; i < N; i++) {
        h_depth_cpu[i] = INF_CPU;
    }
    
    std::queue<int> q;
    h_depth_cpu[source] = 0;
    q.push(source);
    
    while (!q.empty()) {
        int v = q.front();
        q.pop();
        int depth_v = h_depth_cpu[v];
        
        int row_start = h_row[v];
        int row_end = h_row[v + 1];
        
        for (int e = row_start; e < row_end; e++) {
            int u = h_col[e];
            int new_depth = depth_v + 1;
            
            if (new_depth < h_depth_cpu[u]) {
                h_depth_cpu[u] = new_depth;
                q.push(u);
            }
        }
    }
    
    // Compare results
    int error_count = 0;
    int total_error = 0;
    int max_errors_to_print = 20;
    
    for (int i = 0; i < N; i++) {
        int diff = abs(h_depth_cpu[i] - h_depth[i]);
        total_error += diff;
        
        if (h_depth_cpu[i] != h_depth[i]) {
            error_count++;
            if (error_count <= max_errors_to_print) {
                printf("ERROR: depth[%d] mismatch: CPU=%d, GPU=%d\n", 
                       i, h_depth_cpu[i], h_depth[i]);
            }
        }
    }
    
    printf("Validation results:\n");
    printf("  Total vertices: %d\n", N);
    printf("  Errors found: %d\n", error_count);
    printf("  Total error sum: %d\n", total_error);
    
    if (error_count == 0) {
        printf("  ✓ BFS results are CORRECT!\n");
    } else {
        printf("  ✗ BFS results have ERRORS!\n");
    }
    
    // Print first 20 depths for comparison
    printf("\nFirst 20 depths comparison:\n");
    printf("CPU: ");
    for (int i = 0; i < 20 && i < N; i++) {
        printf("%d ", h_depth_cpu[i]);
    }
    printf("\nGPU: ");
    for (int i = 0; i < 20 && i < N; i++) {
        printf("%d ", h_depth[i]);
    }
    printf("\n");
    
    free(h_depth_cpu);

    #ifdef PROFILE
    visualize_profile("bfs");
    #endif

    #ifdef COUNT_TASKS
    unsigned long long h_task_num = 0;
    cudaMemcpyFromSymbol(&h_task_num, g_task_num, sizeof(unsigned long long));
    printf("task_num=%llu\n", h_task_num);
    #endif
    free(h_row); free(h_col); free(h_depth);
    cudaFree(d_row); cudaFree(d_col); cudaFree(d_depth);
    return 0;
}

// #include <stdio.h>
// #include <cuda_runtime.h>
// #include <fstream>
// #include <string>

// #define PROFILE
// #define MAX_PROFILE_DATA 20000

// #define NUM_BLOCKS 3000
// #define THREADS_PER_BLK 32
// #define MAX_TASKS_PER_WARP 100000
// #define INNER_QUEUE_SIZE 64
// #define MAX_CHILD_TASKS 10000

// #define UNNECESSARY_JOIN

// // #define COUNT_TASKS

// #define BATCH_SIZE 100

// struct Task {
//     int v;           // vertex ID
//     int edge_start;  // start index in g_col_indices (inclusive)
//     int edge_end;    // end index in g_col_indices (exclusive)
// };

// // Device-side graph state
// __device__ int* g_row_offsets;   // size: num_vertices + 1
// __device__ int* g_col_indices;   // size: num_edges
// __device__ int* g_depth;         // size: num_vertices; INF indicates unvisited
// __device__ int  g_num_vertices;  // number of vertices
// #ifdef COUNT_TASKS
// __device__ unsigned long long  g_task_num;      // number of tasks
// #endif

// // Include simple task runtime after Task is defined
// #include "task_api_all.cuh"

// // Helper to chunk spawns to respect MAX_CHILD_TASKS
// TASK_DEF(bfs, Task) {
//     TASK_BEGIN();
//     #ifdef COUNT_TASKS
//     atomicAdd(&g_task_num, 1ull);
//     #endif

//     const int v = self->v;
//     int dv = g_depth[v];
    
//     // Process edges in the specified range [edge_start, edge_end)
//     int edge_start = self->edge_start;
//     int edge_end = self->edge_end;
    
//     // Process edges in the current batch
//     for (int e = edge_start; e < edge_end; ++e) {
//         int u = g_col_indices[e];
//         int old = atomicMin(&g_depth[u], dv + 1);
//         if (old > dv + 1) {
//             // For child tasks, we need to process all edges of vertex u
//             // Split into batches of BATCH_SIZE
//             int u_row_start = g_row_offsets[u];
//             int u_row_end = g_row_offsets[u + 1];
            
//             for (int batch_start = u_row_start; batch_start < u_row_end; batch_start += BATCH_SIZE) {
//                 int batch_end = min(batch_start + BATCH_SIZE, u_row_end);
//                 Task child{u, batch_start, batch_end};
//                 TASK_SPAWN(child, bfs, 0);
//             }
//         }
//     }
//     TASK_FINISH();
//     TASK_END();
// }

// __global__ void enqueue_initial_task(int source) {
//     // Root depth = 0 (assume host initialized INF elsewhere)
//     g_depth[source] = 0;
//     // For initial task, process first batch of edges
//     // Remaining batches will be spawned when this task executes
//     int row_start = g_row_offsets[source];
//     int row_end = g_row_offsets[source + 1];
//     Task initial{source, row_start, row_end};
//     TASK_ENQUEUE_INITIAL(Task, bfs, initial);
// }

// // Simple demo host: build a chain graph 0->1->2->...->(N-1)
// static void build_chain_graph(int N, int** h_row, int** h_col, int* M_out) {
//     int* row = (int*)malloc(sizeof(int) * (N + 1));
//     int* col = (int*)malloc(sizeof(int) * (N - 1));
//     for (int i = 0; i <= N; ++i) row[i] = i < N ? i : (N - 1);
//     for (int i = 0; i < N - 1; ++i) col[i] = i + 1;
//     *h_row = row;
//     *h_col = col;
//     *M_out = N - 1;
// }

// static bool read_csr_binary(const char* path, int** h_row, int** h_col, int* N_out, int* M_out) {
//     std::ifstream fin(path, std::ios::binary);
//     if (!fin.is_open()) return false;
//     int N = 0, M = 0;
//     fin.read(reinterpret_cast<char*>(&N), sizeof(int));
//     fin.read(reinterpret_cast<char*>(&M), sizeof(int));
//     if (!fin) return false;
//     int* row = (int*)malloc(sizeof(int) * (N + 1));
//     int* col = (int*)malloc(sizeof(int) * M);
//     if (row == nullptr || col == nullptr) return false;
//     fin.read(reinterpret_cast<char*>(row), sizeof(int) * (N + 1));
//     fin.read(reinterpret_cast<char*>(col), sizeof(int) * M);
//     if (!fin) { free(row); free(col); return false; }
//     *h_row = row; *h_col = col; *N_out = N; *M_out = M;
//     return true;
// }

// int main(int argc, char** argv) {
//     std::string csr_path;
//     int N = 1000; // default chain graph vertices
//     int source = 0;
//     if (argc >= 2) csr_path = argv[1];
//     if (argc >= 3) source = atoi(argv[2]);

//     cudaSetDevice(0);
//     cudaError_t st_init = init_task_runtime<Task>();
//     if (st_init != cudaSuccess) {
//         fprintf(stderr, "init_task_runtime failed: %s\n", cudaGetErrorString(st_init));
//         return 1;
//     }

//     int *h_row = nullptr, *h_col = nullptr; int M = 0;
//     if (!csr_path.empty()) {
//         if (!read_csr_binary(csr_path.c_str(), &h_row, &h_col, &N, &M)) {
//             fprintf(stderr, "Failed to read CSR file: %s\n", csr_path.c_str());
//             return 1;
//         }
//         fprintf(stdout, "Loaded CSR: N=%d, M=%d from %s\n", N, M, csr_path.c_str());
//         fflush(stdout);
//     } else {
//         build_chain_graph(N, &h_row, &h_col, &M);
//         fprintf(stdout, "Using synthetic chain graph: N=%d, M=%d\n", N, M);
//     }

//     int *d_row = nullptr, *d_col = nullptr, *d_depth = nullptr;
//     cudaMalloc(&d_row, sizeof(int) * (N + 1));
//     cudaMalloc(&d_col, sizeof(int) * M);
//     cudaMalloc(&d_depth, sizeof(int) * N);
//     cudaMemcpy(d_row, h_row, sizeof(int) * (N + 1), cudaMemcpyHostToDevice);
//     cudaMemcpy(d_col, h_col, sizeof(int) * M, cudaMemcpyHostToDevice);
//     cudaMemset(d_depth, 0x3f, sizeof(int) * N);

//     // Bind device pointers to device symbols
//     cudaMemcpyToSymbol(g_row_offsets, &d_row, sizeof(d_row));
//     cudaMemcpyToSymbol(g_col_indices, &d_col, sizeof(d_col));
//     cudaMemcpyToSymbol(g_depth, &d_depth, sizeof(d_depth));
//     cudaMemcpyToSymbol(g_num_vertices, &N, sizeof(int));
//     #ifdef COUNT_TASKS
//     unsigned long long zero_task_num = 0ull;
//     cudaMemcpyToSymbol(g_task_num, &zero_task_num, sizeof(unsigned long long));
//     #endif

//     // Timing
//     cudaEvent_t start, stop;
//     cudaEventCreate(&start);
//     cudaEventCreate(&stop);
//     cudaEventRecord(start);

//     enqueue_initial_task<<<1, 1>>>(source);
//     execute_task_loop<TERMINATE_ON_ALL_TASKS_FINISH, Task><<<NUM_BLOCKS, THREADS_PER_BLK>>>();

//     cudaEventRecord(stop);
//     cudaEventSynchronize(stop);
//     float ms = 0.0f; cudaEventElapsedTime(&ms, start, stop);

//     int* h_depth = (int*)malloc(sizeof(int) * N);
//     cudaMemcpy(h_depth, d_depth, sizeof(int) * N, cudaMemcpyDeviceToHost);
//     printf("BFS done. N=%d, source=%d, time=%.3f ms\n", N, source, ms);
//     printf("depth[source]=%d, depth[%d]=%d, depth[%d]=%d, depth[%d]=%d, depth[%d]=%d, depth[%d]=%d\n",
//            h_depth[source], 1, h_depth[1], 2, h_depth[2], 3, h_depth[3], 4, h_depth[4], 5, h_depth[5]);

//     #ifdef PROFILE
//     visualize_profile("bfs");
//     #endif

//     #ifdef COUNT_TASKS
//     unsigned long long h_task_num = 0;
//     cudaMemcpyFromSymbol(&h_task_num, g_task_num, sizeof(unsigned long long));
//     printf("task_num=%llu\n", h_task_num);
//     #endif
//     free(h_row); free(h_col); free(h_depth);
//     cudaFree(d_row); cudaFree(d_col); cudaFree(d_depth);
//     return 0;
// }
