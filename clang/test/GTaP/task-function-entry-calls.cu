// RUN: %clang_cc1 -triple nvptx64-nvidia-cuda -fcuda-is-device -x cuda \
// RUN:   -std=c++17 -fsyntax-only -verify %s

#define __device__ __attribute__((device))
#define __global__ __attribute__((global))

namespace gtap::detail::thread {
struct TaskContext {};
}
using TaskContext = gtap::detail::thread::TaskContext;
struct uint3 { unsigned x, y, z; };
extern __device__ const uint3 blockIdx;
extern __device__ const uint3 threadIdx;
using TaskFn = void (*)(void *, int, TaskContext *);
__device__ void *__gtap_spawn_task(TaskContext *, int, int *, TaskFn, int);
__device__ void __gtap_finish_task(int, TaskContext *);
__device__ int __gtap_get_task_state(int);
__device__ bool __gtap_set_state_for_join(int, int, int, int);
__device__ void *__gtap_get_task_data(int);
__device__ void __gtap_push_initial_task(TaskFn, int);
__device__ void __gtap_execute_task_loop();

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

__device__ int ordinary_entry(int x) { return x; }

__global__ void invalid_entry_not_task_function() {
  // expected-error@+1 {{#pragma gtap entry must be followed by a direct call to a GTaP task function or an assignment from such a call}}
#pragma gtap entry
  ordinary_entry(1);
}
