// RUN: %clang_cc1 -triple nvptx64-nvidia-cuda -fcuda-is-device -x cuda \
// RUN:   -std=c++17 -fsyntax-only %s

#define __device__ __attribute__((device))

struct TaskContext {};
constexpr unsigned long long __gtap_max_task_size = ~0ULL;
using TaskFn = void (*)(void *, int, TaskContext *);
__device__ void *__gtap_spawn_task(TaskContext *, int, int *, TaskFn, int);
__device__ void __gtap_finish_task(int, TaskContext *);
__device__ int __gtap_get_task_state(int);
__device__ bool __gtap_set_state_for_join(int, int, int, int);

#pragma gtap function
__device__ int leaf(int x) { return x; }

#pragma gtap function
__device__ int control_flow(int n) {
  int total = 0;
  for (int i = 0; i < n; ++i) {
    if (i & 1) {
#pragma gtap task
      total = leaf(i);
#pragma gtap taskwait
    } else {
      int j = 0;
      while (j++ < 1) {
#pragma gtap task
        total = leaf(i + 1);
#pragma gtap taskwait
      }
    }
    if (total < 0)
      return total;
  }
  return total;
}
