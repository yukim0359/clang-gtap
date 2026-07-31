// RUN: %clang_cc1 -triple nvptx64-nvidia-cuda -fcuda-is-device -x cuda \
// RUN:   -std=c++17 -fsyntax-only -verify %s

#define __device__ __attribute__((device))
#define __global__ __attribute__((global))

struct TaskContext {};
struct uint3 { unsigned x, y, z; };
extern __device__ const uint3 blockIdx;
extern __device__ const uint3 threadIdx;
constexpr unsigned long long __gtap_max_task_size = ~0ULL;
using TaskFn = void (*)(void *, int, TaskContext *);
__device__ void *__gtap_spawn_task(TaskContext *, int, int *, TaskFn, int);
__device__ void __gtap_finish_task(int, TaskContext *);
__device__ int __gtap_get_task_state(int);
__device__ bool __gtap_set_state_for_join(int, int, int, int);
__device__ void *__gtap_get_task_data(int);
__device__ void __gtap_push_initial_task(TaskFn, int);
__device__ void __gtap_execute_task_loop_device();

#pragma gtap function
__device__ int entry_child(int x) { return x + 1; }

__global__ void valid_entry() {
#pragma gtap entry
  entry_child(1);
}

__global__ void invalid_nested_entry() {
#pragma gtap entry
  // expected-error@+1 {{direct call to GTaP task function 'entry_child' is not supported; use '#pragma gtap task' to spawn it}}
  entry_child(entry_child(1));
}
