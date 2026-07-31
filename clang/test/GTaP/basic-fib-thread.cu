// RUN: %clang_cc1 -triple nvptx64-nvidia-cuda -fcuda-is-device -x cuda \
// RUN:   -std=c++17 -ast-dump %s | FileCheck %s
// CHECK: RecordDecl {{.*}} struct fib_task_data definition
// CHECK: FieldDecl {{.*}} n 'int'
// CHECK: FieldDecl {{.*}} __cap_x 'int'
// CHECK: FieldDecl {{.*}} __cap_y 'int'
// CHECK: FieldDecl {{.*}} __gtap_result 'int'
// CHECK: FieldDecl {{.*}} __gtap_result_dst 'int *'
// CHECK: FunctionDecl {{.*}} __gtap_state_machine_fib
// CHECK-COUNT-2: DeclRefExpr {{.*}} Function {{.*}} '__gtap_spawn_task'
// CHECK: DeclRefExpr {{.*}} Function {{.*}} '__gtap_set_state_for_join'
// CHECK: DeclRefExpr {{.*}} Function {{.*}} '__gtap_finish_task'

#define __device__ __attribute__((device))

struct TaskContext {};
constexpr unsigned long long __gtap_max_task_size = ~0ULL;
using TaskFn = void (*)(void *, int, TaskContext *);
__device__ void *__gtap_spawn_task(TaskContext *, int, int *, TaskFn, int);
__device__ void __gtap_finish_task(int, TaskContext *);
__device__ int __gtap_get_task_state(int);
__device__ bool __gtap_set_state_for_join(int, int, int, int);

#pragma gtap function
__device__ int fib(int n) {
  if (n < 2)
    return n;
  int x;
  int y;
#pragma gtap task
  x = fib(n - 1);
#pragma gtap task
  y = fib(n - 2);
#pragma gtap taskwait
  return x + y;
}
