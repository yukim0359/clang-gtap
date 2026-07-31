// RUN: %clang_cc1 -triple nvptx64-nvidia-cuda -fcuda-is-device -x cuda \
// RUN:   -std=c++17 -ast-dump %s | FileCheck %s
// CHECK: RecordDecl {{.*}} struct two_waits_task_data definition
// CHECK: FieldDecl {{.*}} __cap_a 'int'
// CHECK: FieldDecl {{.*}} __cap_b 'int'
// CHECK: FunctionDecl {{.*}} __gtap_state_machine_two_waits
// CHECK-COUNT-2: DeclRefExpr {{.*}} Function {{.*}} '__gtap_set_state_for_join'

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
__device__ int two_waits(int x) {
  int a;
  int b;
#pragma gtap task
  a = leaf(x);
#pragma gtap taskwait
#pragma gtap task
  b = leaf(a + 1);
#pragma gtap taskwait
  return a + b;
}
