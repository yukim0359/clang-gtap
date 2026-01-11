#include <cstdio>
#include <cstdlib>
#include <vector>
#include <random>
#include <algorithm>
#include <limits>
#include <cuda_runtime.h>

// #define PROFILE

#ifndef SCALING_EVALUATION
#define TASK_KIND 1

#define NUM_BLOCKS 3000
#define THREADS_PER_BLK 64
#define MAX_TASKS_PER_WARP 100000
#define MAX_CHILD_TASKS 7
#endif

// #define MAX_PROFILE_DATA 30000

#define TASK_SPAWN_CUTOFF_SORT 64
#define TASK_SPAWN_CUTOFF_MERGE 256

#ifdef USE_GQ
#include "task_user_api_gq.cuh"
#else
#include "task_api_all.cuh"
#endif

struct Task {
    // cilksort
    int* arr;
    int n;
    int* tmp;
    // cilkmerge
    int* a;
    int a_len;
    int* b;
    int b_len;
    int* dst;
};

// Global device pointers to data and buffer
__device__ int* g_data;
__device__ int* g_buf;

// Sequential merge
__device__ __forceinline__ void merge_device(int* a, int a_len, int* b, int b_len, int* dst) {
    int i = 0, j = 0, ti = 0;
    while (i < a_len && j < b_len) {
        int a_val = load_L2(&a[i]);
        int b_val = load_L2(&b[j]);
        if (a_val < b_val) {
            // store_L2(&dst[ti++], a_val);
            dst[ti++] = a_val;
            i++;
        } else {
            // store_L2(&dst[ti++], b_val);
            dst[ti++] = b_val;
            j++;
        }
    }
    while (i < a_len) {
        // store_L2(&dst[ti++], load_L2(&a[i++]));
        dst[ti++] = load_L2(&a[i++]);
    }
    while (j < b_len) {
        // store_L2(&dst[ti++], load_L2(&b[j++]));
        dst[ti++] = load_L2(&b[j++]);
    }
}

// Binary search: return the largest index s.t. x[index] <= val
__device__ __forceinline__ int binary_search_device(int* x, int len, int val) {
    int low = -1;
    int high = len;
    while (low + 1 < high) {
        int mid = (low + high) / 2;
        int mid_val = load_L2(&x[mid]);
        if (mid_val <= val) low = mid;
        else high = mid;
    }
    return low;
}

// Sequential sort (using insertion sort for small arrays)
// __device__ __forceinline__ void sort_small_device(int* arr, int n) {
//     for (int i = 1; i < n; i++) {
//         int key = load_L2(&arr[i]);
//         int j = i - 1;
//         while (j >= 0 && load_L2(&arr[j]) > key) {
//             int arr_j = load_L2(&arr[j]);
//             // store_L2(&arr[j + 1], arr_j);
//             arr[j + 1] = arr_j;
//             j--;
//         }
//         // store_L2(&arr[j + 1], key);
//         arr[j + 1] = key;
//     }
// }

__device__ __forceinline__ void sort_small_device(int* arr, int n) {
    int buf[TASK_SPAWN_CUTOFF_SORT];

    #pragma unroll
    for (int i = 0; i < TASK_SPAWN_CUTOFF_SORT; ++i) {
        if (i < n) {
            buf[i] = load_L2(&arr[i]);
        } else {
            buf[i] = INT_MAX;  // 昇順ソートなので末尾に飛ばす
        }
    }

    for (int k = 2; k <= TASK_SPAWN_CUTOFF_SORT; k <<= 1) {
        for (int j = k >> 1; j > 0; j >>= 1) {
            #pragma unroll
            for (int i = 0; i < TASK_SPAWN_CUTOFF_SORT; ++i) {
                int ixj = i ^ j;
                if (ixj > i) {
                    bool up = ((i & k) == 0);  // 上向き or 下向き

                    int a = buf[i];
                    int b = buf[ixj];

                    // 比較結果で swap するかどうかだけがデータ依存
                    bool cond = up ? (a > b) : (a < b);
                    if (cond) {
                        buf[i]   = b;
                        buf[ixj] = a;
                    }
                }
            }
        }
    }

    #pragma unroll
    for (int i = 0; i < n; ++i) {
        arr[i] = buf[i];
    }
}

// Parallel merge task
TASK_DEF(merge_task, Task) {
    int* a;
    int* b;
    int* dst;
    int a_len, b_len;
    int a_split, b_split;
    TASK_BEGIN();
    // printf("merge_first_task: a_left = %d, a_right = %d, a_len = %d, b_left = %d, b_right = %d, b_len = %d\n", (int)(self->a - g_data), (int)(self->a - g_data) + self->a_len, self->a_len, (int)(self->b - g_data), (int)(self->b - g_data) + self->b_len, self->b_len);
    // Read task data fields atomically
    a = reinterpret_cast<int*>(load_task_field_ptr(self, offsetof(Task, a)));
    a_len = load_task_field_int(self, offsetof(Task, a_len));
    b = reinterpret_cast<int*>(load_task_field_ptr(self, offsetof(Task, b)));
    b_len = load_task_field_int(self, offsetof(Task, b_len));
    dst = reinterpret_cast<int*>(load_task_field_ptr(self, offsetof(Task, dst)));

    // Ensure a_len >= b_len
    if (a_len < b_len) {
        int* tmp = a; a = b; b = tmp;
        int tmp2 = a_len; a_len = b_len; b_len = tmp2;
    }

    // printf("merge_task: a_len = %d\n", a_len);
    // printf("merge_task: b_len = %d\n", b_len);

    // Base case: empty arrays
    if (b_len == 0) {
        // Copy a to dst if a_len > 0
        if (a_len > 0) {
            // for (int i = 0; i < a_len; i++) {
            //     dst[i] = a[i];
            // }
            // printf("merge_task: a_len = %d\n", a_len);
            // printf("merge_task: b_len = %d\n", b_len);
            // memcpy(dst, a, a_len * sizeof(int));
            for (int i = 0; i < a_len; i++) {
                // store_L2(&dst[i], load_L2(&a[i]));
                dst[i] = load_L2(&a[i]);
            }
            printf("merge_task: a_len = %d\n", a_len);
        }
        __threadfence();
        TASK_FINISH();
    }

    // Sequential merge for small arrays
    if (a_len + b_len < TASK_SPAWN_CUTOFF_MERGE) {
        merge_device(a, a_len, b, b_len, dst);
        TASK_FINISH();
    }

    // Parallel merge: split and recurse
    a_split = (a_len + 1) / 2;
    b_split = binary_search_device(b, b_len, load_L2(&a[a_split - 1])) + 1;

    Task left_task;
    left_task.a = a;
    left_task.a_len = a_split;
    left_task.b = b;
    left_task.b_len = b_split;
    left_task.dst = dst;

    Task right_task;
    right_task.a = a + a_split;
    right_task.a_len = a_len - a_split;
    right_task.b = b + b_split;
    right_task.b_len = b_len - b_split;
    right_task.dst = dst + a_split + b_split;

    TASK_SPAWN(left_task, merge_task, 0);
    TASK_SPAWN(right_task, merge_task, 0);
    TASK_JOIN(0);
    TASK_FINISH();
    TASK_END();
}

// Sort task
TASK_DEF(sort_task, Task) {
    int* arr;
    int* tmp;
    int n;
    int len12, len1, len2, len34, len3, len4;
    TASK_BEGIN();
    // printf("sort_task: left = %d, right = %d\n", (int)(self->arr - g_data), (int)(self->arr - g_data) + self->n);
    // Read task data fields atomically
    arr = reinterpret_cast<int*>(load_task_field_ptr(self, offsetof(Task, arr)));
    n = load_task_field_int(self, offsetof(Task, n));
    tmp = reinterpret_cast<int*>(load_task_field_ptr(self, offsetof(Task, tmp)));

    if (n < 2) TASK_FINISH();

    // Sequential sort for small arrays
    if (n < TASK_SPAWN_CUTOFF_SORT) {
        sort_small_device(arr, n);
        TASK_FINISH();
    }

    // Cilksort: divide into 4 parts
    len12 = n / 2;
    len1 = len12 / 2;
    len2 = len12 - len1;
    len34 = n - len12;
    len3 = len34 / 2;
    len4 = len34 - len3;

    // Spawn 4 sort tasks
    Task task1;
    task1.arr = arr;
    task1.n = len1;
    task1.tmp = tmp;

    Task task2;
    task2.arr = arr + len1;
    task2.n = len2;
    task2.tmp = tmp + len1;

    Task task3;
    task3.arr = arr + len12;
    task3.n = len3;
    task3.tmp = tmp + len12;

    Task task4;
    task4.arr = arr + len12 + len3;
    task4.n = len4;
    task4.tmp = tmp + len12 + len3;

    TASK_SPAWN(task1, sort_task, 0);
    TASK_SPAWN(task2, sort_task, 0);
    TASK_SPAWN(task3, sort_task, 0);
    TASK_SPAWN(task4, sort_task, 0);
    TASK_JOIN(0);

    // Read task data fields atomically
    arr = reinterpret_cast<int*>(load_task_field_ptr(self, offsetof(Task, arr)));
    n = load_task_field_int(self, offsetof(Task, n));
    tmp = reinterpret_cast<int*>(load_task_field_ptr(self, offsetof(Task, tmp)));
    len12 = n / 2;
    len1 = len12 / 2;
    len2 = len12 - len1;
    len34 = n - len12;
    len3 = len34 / 2;
    len4 = len34 - len3;

    Task merge1;
    merge1.a = arr;
    merge1.a_len = len1;
    merge1.b = arr + len1;
    merge1.b_len = len2;
    merge1.dst = tmp;

    Task merge2;
    merge2.a = arr + len12;
    merge2.a_len = len3;
    merge2.b = arr + len12 + len3;
    merge2.b_len = len4;
    merge2.dst = tmp + len12;

    TASK_SPAWN(merge1, merge_task, 0);
    TASK_SPAWN(merge2, merge_task, 0);
    TASK_JOIN(0);

    // Read task data fields atomically
    arr = reinterpret_cast<int*>(load_task_field_ptr(self, offsetof(Task, arr)));
    n = load_task_field_int(self, offsetof(Task, n));
    tmp = reinterpret_cast<int*>(load_task_field_ptr(self, offsetof(Task, tmp)));
    len12 = n / 2;
    len1 = len12 / 2;
    len2 = len12 - len1;
    len34 = n - len12;
    len3 = len34 / 2;
    len4 = len34 - len3;

    Task final_merge;
    final_merge.a = tmp;
    final_merge.a_len = len12;
    final_merge.b = tmp + len12;
    final_merge.b_len = len34;
    final_merge.dst = arr;
    
    TASK_SPAWN(final_merge, merge_task, 0);
    TASK_JOIN(0);
    TASK_FINISH();
    TASK_END();
}

// Enqueue initial task
__global__ void enqueue_initial_task(int n) {
    Task initial;
    initial.arr = g_data;
    initial.n = n;
    initial.tmp = g_buf;
    TASK_ENQUEUE_INITIAL(Task, sort_task, initial);
}

// Helpers to bind device pointers
static inline cudaError_t bind_device_arrays(int* d_data, int* d_buf) {
    cudaError_t st;
    st = cudaMemcpyToSymbol(g_data, &d_data, sizeof(d_data)); if (st != cudaSuccess) return st;
    st = cudaMemcpyToSymbol(g_buf,  &d_buf,  sizeof(d_buf));  if (st != cudaSuccess) return st;
    return cudaSuccess;
}

// Load array from binary file
std::vector<int> load_array(const char* filename, size_t& n) {
    std::vector<int> data;
    FILE* fp = fopen(filename, "rb");
    if (!fp) {
        fprintf(stderr, "Error: Cannot open %s for reading\n", filename);
        return data;
    }
    
    // Read size
    if (fread(&n, sizeof(size_t), 1, fp) != 1) {
        fprintf(stderr, "Error: Cannot read size from %s\n", filename);
        fclose(fp);
        return data;
    }
    
    // Allocate and read data
    data.resize(n);
    if (fread(data.data(), sizeof(int), n, fp) != n) {
        fprintf(stderr, "Error: Cannot read data from %s\n", filename);
        data.clear();
        fclose(fp);
        return data;
    }
    
    fclose(fp);
    printf("Loaded %zu elements from %s\n", n, filename);
    return data;
}

int main(int argc, char** argv) {
    printf("size of Task: %zu bytes\n", sizeof(Task));
    if (argc < 2) {
        printf("Usage: %s <data_file> [reference_file]\n", argv[0]);
        printf("  data_file: Binary file containing array data (generated by gen_vector)\n");
        printf("  reference_file: Optional reference file for verification\n");
        return 1;
    }
    
    const char* data_file = argv[1];
    const char* reference_file = (argc > 2) ? argv[2] : nullptr;
    
    cudaSetDevice(0);
   
    // // Increase printf buffer size to avoid truncation
    // // Default is ~1MB, increase to 100MB for large outputs
    // size_t printf_fifo_size = 100 * 1024 * 1024; // 100MB
    // cudaError_t st = cudaDeviceSetLimit(cudaLimitPrintfFifoSize, printf_fifo_size);
    // if (st != cudaSuccess) {
    //     fprintf(stderr, "Warning: Failed to set printf buffer size: %s\n", cudaGetErrorString(st));
    // }
    
    // st = init_task_runtime<Task>();

    cudaError_t st = init_task_runtime<Task>();
    if (st != cudaSuccess) {
        fprintf(stderr, "init_task_runtime failed: %s\n", cudaGetErrorString(st));
        return 1;
    }

    // Load data from file
    size_t N;
    std::vector<int> h = load_array(data_file, N);
    if (h.empty()) return 1;
    
    // Create reference for verification
    std::vector<int> gold;
    if (reference_file) {
        size_t ref_n;
        gold = load_array(reference_file, ref_n);
        if (gold.empty() || ref_n != N) {
            fprintf(stderr, "Error: Reference file size mismatch\n");
            return 1;
        }
    } else {
        gold = h;
        std::sort(gold.begin(), gold.end());
    }

    // Device buffers
    int* d_data = nullptr; int* d_buf = nullptr;
    cudaMalloc(&d_data, sizeof(int) * N);
    cudaMalloc(&d_buf,  sizeof(int) * N);
    cudaMemcpy(d_data, h.data(), sizeof(int) * N, cudaMemcpyHostToDevice);
    bind_device_arrays(d_data, d_buf);

    // Run
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    enqueue_initial_task<<<1, 1>>>(static_cast<int>(N));
    execute_task_loop<TERMINATE_ON_FIRST_TASK_FINISH, Task><<<NUM_BLOCKS, THREADS_PER_BLK>>>();

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    // // Ensure all printf output is flushed
    // cudaDeviceSynchronize();
    // fflush(stdout);
    // fflush(stderr);
    float ms = 0.f;
    cudaEventElapsedTime(&ms, start, stop);

    // Copy back and verify
    cudaMemcpy(h.data(), d_data, sizeof(int) * N, cudaMemcpyDeviceToHost);
    std::sort(gold.begin(), gold.end());
    bool ok = std::is_sorted(h.begin(), h.end()) && (h == gold);

    // Match mergesort.cu output style
    printf("Cilksort(%zu) = %s\n", N, ok ? "Correct" : "Incorrect");
    printf("Kernel execution time: %.3f ms (%.6f s)\n", ms, ms / 1000.0);

    #ifdef PROFILE
    visualize_profile("cilksort");
    #endif

    cudaFree(d_data); cudaFree(d_buf);
    return ok ? 0 : 1;
}
