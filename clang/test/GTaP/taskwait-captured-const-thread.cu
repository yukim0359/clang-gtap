// RUN: %clang_cc1 -triple nvptx64-nvidia-cuda -fcuda-is-device -x cuda \
// RUN:   -std=c++17 -ast-dump %s | FileCheck %s
// Storage drops top-level const so the initializer can be stored.
// Pointee const stays.
// CHECK: FieldDecl {{.*}} input 'const int *'
// CHECK: FieldDecl {{.*}} __cap_saved 'int'
// CHECK: FieldDecl {{.*}} __cap_pinned 'const int *'
// CHECK: FunctionDecl {{.*}} __gtap_state_machine_keep_const
// CHECK: BinaryOperator {{.*}} 'int' lvalue '='
// CHECK-NEXT: MemberExpr {{.*}}__cap_saved
// CHECK: BinaryOperator {{.*}} 'const int *' lvalue '='
// CHECK-NEXT: MemberExpr {{.*}}__cap_pinned

#define __device__ __attribute__((device))

namespace gtap::detail::thread {
struct TaskContext {};
}
using TaskContext = gtap::detail::thread::TaskContext;
using TaskFn = void (*)(void *, int, TaskContext *);
__device__ void *__gtap_spawn_task(TaskContext *, int, int *, TaskFn, int);
__device__ void __gtap_finish_task(int, TaskContext *);
__device__ int __gtap_get_task_state(int);
__device__ bool __gtap_set_state_for_join(int, int, int, int);

#pragma gtap function
__device__ int keep_const(const int *input) {
  const int saved = *input;
  const int *const pinned = input;
#pragma gtap taskwait
  return saved + *pinned;
}
