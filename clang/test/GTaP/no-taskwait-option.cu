// RUN: %clang_cc1 -triple nvptx64-nvidia-cuda -fcuda-is-device -x cuda \
// RUN:   -std=c++17 -fgtap-no-taskwait -fsyntax-only -verify %s

#define __device__ __attribute__((device))

struct TaskContext {};
constexpr unsigned long long __gtap_max_task_size = ~0ULL;
__device__ void __gtap_finish_task(int, TaskContext *);
__device__ int __gtap_get_task_state(int);
__device__ bool __gtap_set_state_for_join(int, int, int, int);

#pragma gtap function
__device__ void forbidden_wait() {
  // expected-error@+1 {{#pragma gtap taskwait cannot be used with -fgtap-no-taskwait or GTAP_ASSUME_NO_TASKWAIT}}
#pragma gtap taskwait
}
