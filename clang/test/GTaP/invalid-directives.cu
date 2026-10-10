// RUN: %clang_cc1 -triple nvptx64-nvidia-cuda -fcuda-is-device -x cuda \
// RUN:   -std=c++17 -fsyntax-only -verify %s

#define __device__ __attribute__((device))

namespace gtap::detail::thread {
struct TaskContext {};
}
using TaskContext = gtap::detail::thread::TaskContext;
using TaskFn = void (*)(void *, int, TaskContext *);
__device__ void *__gtap_spawn_task(TaskContext *, int, int *, TaskFn, int);
__device__ void __gtap_finish_task(int, TaskContext *);
__device__ int __gtap_get_task_state(int);
__device__ bool __gtap_prepare_for_join(int, int, int, int);

__device__ void ordinary();

#pragma gtap function
__device__ void queue_leaf() {}

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
__device__ void invalid_queue(int *p, float f) {
  // expected-error@+1 {{queue argument of type 'int *' is not an integer}}
#pragma gtap task queue(p)
  queue_leaf();
  // expected-error@+1 {{queue argument of type 'float' is not an integer}}
#pragma gtap taskwait queue(f)
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
