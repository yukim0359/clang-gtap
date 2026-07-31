// RUN: %clang_cc1 -triple nvptx64-nvidia-cuda -fcuda-is-device -x cuda \
// RUN:   -std=c++17 -fsyntax-only -verify %s

#define __device__ __attribute__((device))
#define __global__ __attribute__((global))

struct TaskContext {};
constexpr unsigned long long __gtap_max_task_size = ~0ULL;
using TaskFn = void (*)(void *, int, TaskContext *);
__device__ void *__gtap_spawn_task(TaskContext *, int, int *, TaskFn, int);
__device__ void __gtap_finish_task(int, TaskContext *);
__device__ int __gtap_get_task_state(int);
__device__ bool __gtap_set_state_for_join(int, int, int, int);

// The pragma may annotate a declaration; the later definition inherits it.
#pragma gtap function
__device__ int child(int);

__device__ int child(int x) { return x + 1; }

__device__ int ordinary_device_function(int x) {
  // expected-error@+1 {{direct call to GTaP task function 'child' is not supported; use '#pragma gtap task' to spawn it}}
  return child(x);
}

__global__ void ordinary_kernel_call() {
  // expected-error@+1 {{direct call to GTaP task function 'child' is not supported; use '#pragma gtap task' to spawn it}}
  child(1);
}

#pragma gtap function
__device__ int overloaded(int);
__device__ int overloaded(int x) { return x; }
__device__ float overloaded(float x) { return x; }

__device__ float overload_calls() {
  // expected-error@+1 {{direct call to GTaP task function 'overloaded' is not supported; use '#pragma gtap task' to spawn it}}
  int task_result = overloaded(1);
  return overloaded(1.0f);
}

#pragma gtap function
__device__ int nested_task_call(int x) {
  int result;
#pragma gtap task
  // expected-error@+1 {{direct call to GTaP task function 'child' is not supported; use '#pragma gtap task' to spawn it}}
  result = child(child(x));
  return result;
}

#pragma gtap function
__device__ int parent(int x) {
  int spawned;
#pragma gtap task
  spawned = child(x);

  // expected-error@+1 {{direct call to GTaP task function 'child' is not supported; use '#pragma gtap task' to spawn it}}
  return child(spawned);
}
