// RUN: %clang_cc1 -triple nvptx64-nvidia-cuda -fcuda-is-device -x cuda \
// RUN:   -std=c++17 -ast-dump %s | FileCheck %s
// A discarded block entry still publishes one result per thread.
// CHECK: VarDecl {{.*}} __gtap_auto_entry_result_size 'const unsigned long' extern cinit
// CHECK-NEXT: IntegerLiteral {{.*}} 'unsigned long' 4
// CHECK: DeclRefExpr {{.*}} Function {{.*}} '__gtap_get_entry_result_data'
// CHECK: MemberExpr {{.*}}__gtap_result_dst
// CHECK: DeclRefExpr {{.*}} '__gtap_entry_result_root_

#define __device__ __attribute__((device))
#define __global__ __attribute__((global))
#define __GTAP_IS_BLOCK_MODE 1

namespace gtap::detail::block {
struct TaskContext {};
}
using TaskContext = gtap::detail::block::TaskContext;
struct uint3 { unsigned x, y, z; };
extern __device__ const uint3 blockIdx;
extern __device__ const uint3 threadIdx;
using TaskFn = void (*)(void *, int, TaskContext *);
__device__ void *__gtap_spawn_task(TaskContext *, int, int *, TaskFn, int);
__device__ void __gtap_finish_task(int, TaskContext *);
__device__ int __gtap_get_task_state(int);
__device__ bool __gtap_set_state_for_join_block(int, TaskContext *, int, int);
__device__ void *__gtap_get_task_data(int);
__device__ void *__gtap_get_entry_result_data();
__device__ void __gtap_push_initial_task(TaskFn, int);
__device__ void __gtap_execute_task_loop();

#pragma gtap function
__device__ int root(int n) { return n; }

__global__ void discarded_entry() {
#pragma gtap entry
  root(1);
}
