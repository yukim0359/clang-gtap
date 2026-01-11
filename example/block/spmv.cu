#include <cstdio>
#include <cstdlib>
#include <vector>
#include <random>
#include <algorithm>
#include <cuda_runtime.h>
#include <fstream>

#include "task_api_all.cuh"

#define RANGE_CUTOFF 16
#define ROWS_PER_BLOCK 8  // Number of rows processed by one block
struct Task {
    int row_idx;
    int begin;
    int end;
};

// Global device pointers to CSR data
__device__ int* g_row_ptr;
__device__ int* g_col_idx;
__device__ double* g_val;
__device__ double* g_x;
__device__ double* g_y;
__device__ int g_n;  // Number of rows

TASK_DEF(spmv_row_task, Task) {
    int b, e;
    TASK_BEGIN();

    b = self->begin;
    e = self->end;

    // Process multiple rows in one block
    // Each thread processes elements for multiple rows
    __shared__ double s_sum[ROWS_PER_BLOCK][THREADS_PER_BLK];
    
    // Initialize shared memory
    for (int r = 0; r < ROWS_PER_BLOCK; r++) {
        if (threadIdx.x < THREADS_PER_BLK) {
            s_sum[r][threadIdx.x] = 0.0;
        }
    }
    __syncthreads();

    // Process each row in the range [b, e)
    for (int row = b; row < e; row++) {
        int row_idx_in_block = row - b;
        if (row_idx_in_block >= ROWS_PER_BLOCK) break;  // Safety check
        
        int row_begin = g_row_ptr[row];
        int row_end   = g_row_ptr[row + 1];

        // Each thread processes elements with stride
        for (int k = row_begin + threadIdx.x; k < row_end; k += blockDim.x) {
            s_sum[row_idx_in_block][threadIdx.x] += g_val[k] * g_x[g_col_idx[k]];
        }
    }

    __syncthreads();

    // Reduce for each row
    for (int row = b; row < e; row++) {
        int row_idx_in_block = row - b;
        if (row_idx_in_block >= ROWS_PER_BLOCK) break;
        
        // Parallel reduction within the block for this row
        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (threadIdx.x < stride) {
                s_sum[row_idx_in_block][threadIdx.x] += s_sum[row_idx_in_block][threadIdx.x + stride];
            }
            __syncthreads();
        }

        if (threadIdx.x == 0) {
            g_y[row] = s_sum[row_idx_in_block][0];
        }
        __syncthreads();
    }

    TASK_FINISH();
    TASK_END();
}

TASK_DEF(range_spmv_task, Task) {
    int b, e;
    int size;
    TASK_BEGIN();

    b = self->begin;
    e = self->end;
    size = e - b;

    if (size <= ROWS_PER_BLOCK) {
        // Process all rows in this range with one task
        if (threadIdx.x == 0) {
            Task t;
            t.begin = b;
            t.end = e;
            TASK_SPAWN(t, spmv_row_task);
        }
    } else {
        // Split range and spawn child tasks
        if (threadIdx.x == 0) {
            int mid = (b + e) >> 1;

            Task left, right;
            left.begin  = b;
            left.end    = mid;
            right.begin = mid;
            right.end   = e;

            TASK_SPAWN(left,  range_spmv_task);
            TASK_SPAWN(right, range_spmv_task);
        }
    }

    TASK_FINISH();
    TASK_END();
}

__global__ void enqueue_initial_task(int n) {
    Task init;
    init.begin = 0;
    init.end   = n;
    TASK_ENQUEUE_INITIAL(Task, range_spmv_task, init);
}

// Helper to bind device pointers
static inline cudaError_t bind_device_arrays(
    int* d_row_ptr, int* d_col_idx, double* d_val,
    double* d_x, double* d_y, int n) {
    cudaError_t st;
    st = cudaMemcpyToSymbol(g_row_ptr, &d_row_ptr, sizeof(d_row_ptr));
    if (st != cudaSuccess) return st;
    st = cudaMemcpyToSymbol(g_col_idx, &d_col_idx, sizeof(d_col_idx));
    if (st != cudaSuccess) return st;
    st = cudaMemcpyToSymbol(g_val, &d_val, sizeof(d_val));
    if (st != cudaSuccess) return st;
    st = cudaMemcpyToSymbol(g_x, &d_x, sizeof(d_x));
    if (st != cudaSuccess) return st;
    st = cudaMemcpyToSymbol(g_y, &d_y, sizeof(d_y));
    if (st != cudaSuccess) return st;
    st = cudaMemcpyToSymbol(g_n, &n, sizeof(n));
    if (st != cudaSuccess) return st;
    return cudaSuccess;
}

// Load CSR matrix from file
bool load_csr_from_file(
    const char* filename,
    int& rows, int& cols, int& nnz,
    std::vector<int>& row_ptr,
    std::vector<int>& col_idx,
    std::vector<double>& val) {
    
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        fprintf(stderr, "Failed to open file %s\n", filename);
        return false;
    }
    
    // Read header: rows, cols, nnz
    file.read(reinterpret_cast<char*>(&rows), sizeof(int));
    file.read(reinterpret_cast<char*>(&cols), sizeof(int));
    file.read(reinterpret_cast<char*>(&nnz), sizeof(int));
    
    if (!file.good()) {
        fprintf(stderr, "Failed to read header from file %s\n", filename);
        file.close();
        return false;
    }
    
    // Allocate and read data
    row_ptr.resize(rows + 1);
    col_idx.resize(nnz);
    val.resize(nnz);
    
    file.read(reinterpret_cast<char*>(row_ptr.data()), (rows + 1) * sizeof(int));
    file.read(reinterpret_cast<char*>(col_idx.data()), nnz * sizeof(int));
    file.read(reinterpret_cast<char*>(val.data()), nnz * sizeof(double));
    
    if (!file.good()) {
        fprintf(stderr, "Failed to read data from file %s\n", filename);
        file.close();
        return false;
    }
    
    file.close();
    return true;
}

int main(int argc, char** argv) {
    const char* filename = "util/spm_data.bin";
    
    if (argc > 1) filename = argv[1];

    // Load CSR matrix from file
    int rows, cols, nnz;
    std::vector<int> row_ptr;
    std::vector<int> col_idx;
    std::vector<double> val;
    
    if (!load_csr_from_file(filename, rows, cols, nnz, row_ptr, col_idx, val)) {
        fprintf(stderr, "Failed to load CSR matrix from file %s\n", filename);
        fprintf(stderr, "Please run gen_spm first to generate the matrix file\n");
        return 1;
    }
    
    printf("Loaded CSR matrix: %d rows, %d cols, %d nnz\n", rows, cols, nnz);

    // Allocate host vectors
    std::vector<double> h_x(cols, 1.0);
    std::vector<double> h_y(rows, 0.0);

    // Allocate device memory
    int* d_row_ptr = nullptr;
    int* d_col_idx = nullptr;
    double* d_val = nullptr;
    double* d_x = nullptr;
    double* d_y = nullptr;

    cudaMalloc(&d_row_ptr, (rows + 1) * sizeof(int));
    cudaMalloc(&d_col_idx, nnz * sizeof(int));
    cudaMalloc(&d_val, nnz * sizeof(double));
    cudaMalloc(&d_x, cols * sizeof(double));
    cudaMalloc(&d_y, rows * sizeof(double));

    // Copy data to device
    cudaMemcpy(d_row_ptr, row_ptr.data(), (rows + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_col_idx, col_idx.data(), nnz * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_val, val.data(), nnz * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, h_x.data(), cols * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, h_y.data(), rows * sizeof(double), cudaMemcpyHostToDevice);

    // Initialize task runtime
    cudaSetDevice(0);
    cudaError_t st = init_task_runtime<Task>();
    if (st != cudaSuccess) {
        fprintf(stderr, "init_task_runtime failed: %s\n", cudaGetErrorString(st));
        return 1;
    }

    // Bind device arrays
    bind_device_arrays(d_row_ptr, d_col_idx, d_val, d_x, d_y, rows);

    // Run SpMV
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    enqueue_initial_task<<<1, 1>>>(rows);
    execute_task_loop<TERMINATE_ON_ALL_TASKS_FINISH, Task><<<NUM_BLOCKS, THREADS_PER_BLK>>>();

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms = 0.f;
    cudaEventElapsedTime(&ms, start, stop);

    // Copy result back
    cudaMemcpy(h_y.data(), d_y, rows * sizeof(double), cudaMemcpyDeviceToHost);

    // Verify result (simple check: sum of y should be reasonable)
    double sum_y = 0.0;
    for (int i = 0; i < rows; ++i) {
        sum_y += h_y[i];
    }
    printf("SpMV(%d rows, %d cols, %d nnz) completed\n", rows, cols, nnz);
    printf("Sum of y: %.6f\n", sum_y);
    printf("Kernel execution time: %.3f ms (%.6f s)\n", ms, ms / 1000.0);

    #ifdef PROFILE
    visualize_profile("spmv");
    #endif

    // Cleanup
    cudaFree(d_row_ptr);
    cudaFree(d_col_idx);
    cudaFree(d_val);
    cudaFree(d_x);
    cudaFree(d_y);

    return 0;
}
