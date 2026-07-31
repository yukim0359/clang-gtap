// RUN: %clang_cc1 -triple nvptx64-nvidia-cuda -fcuda-is-device -x cuda \
// RUN:   -std=c++17 -fsyntax-only -verify %s

#define __device__ __attribute__((device))

struct TaskContext {};
constexpr unsigned long long __gtap_max_task_size = ~0ULL;
using TaskFn = void (*)(void *, int, TaskContext *);
__device__ void *__gtap_spawn_task(TaskContext *, int, int *, TaskFn, int);
__device__ void __gtap_finish_task(int, TaskContext *);
__device__ int __gtap_get_task_state(int);
__device__ bool __gtap_set_state_for_join(int, int, int, int);

__device__ void ordinary();

#pragma gtap function
__device__ void invalid_ordinary_call() {
  // expected-error@+1 {{#pragma gtap task must be followed by a direct call to a GTaP task function or an assignment from such a call}}
#pragma gtap task
  ordinary();
}

#pragma gtap function
__device__ void invalid_non_call(int *x) {
  // expected-error@+1 {{#pragma gtap task must be followed by a direct call to a GTaP task function or an assignment from such a call}}
#pragma gtap task
  ++*x;
}

#pragma gtap function
__device__ void invalid_switch(int x) {
  // expected-error@+1 {{#pragma gtap taskwait inside a switch statement is not supported}}
  switch (x) {
  case 0:
#pragma gtap taskwait
    break;
  }
}
