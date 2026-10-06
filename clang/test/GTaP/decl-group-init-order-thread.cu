// RUN: %clang_cc1 -triple nvptx64-nvidia-cuda -fcuda-is-device -x cuda \
// RUN:   -std=c++17 -ast-dump %s | FileCheck %s
// A mixed declaration keeps source order. saved is captured, so its store
// stays between the earlier and later non-captured initializers.
// CHECK: FunctionDecl {{.*}} __gtap_state_machine_ordered
// CHECK: VarDecl {{.*}} early 'int' cinit
// CHECK: BinaryOperator {{.*}} 'int' lvalue '='
// CHECK-NEXT: MemberExpr {{.*}}__cap_saved
// CHECK: DeclRefExpr {{.*}} 'int' lvalue Var {{.*}} 'early' 'int'
// CHECK: VarDecl {{.*}} temp 'int' cinit
// CHECK-NEXT: BinaryOperator {{.*}} 'int' '+'
// CHECK: MemberExpr {{.*}}__cap_saved
// CHECK: DeclRefExpr {{.*}} 'void (int)' lvalue Function {{.*}} 'use'

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
__device__ void use(int);

#pragma gtap function
__device__ int ordered(int input) {
  int early = 1, saved = early, temp = saved + 1;
  use(early + temp);
#pragma gtap taskwait
  return saved;
}
