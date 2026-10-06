// RUN: %clang_cc1 -triple nvptx64-nvidia-cuda -fcuda-is-device -x cuda \
// RUN:   -std=c++17 -ast-dump %s | FileCheck %s
// input is uniform, so it stays one pointer. Captured locals are per thread,
// and only their top-level const is removed.
// CHECK: FieldDecl {{.*}} input 'const int *'
// CHECK: FieldDecl {{.*}} __cap_saved 'int[32]'
// CHECK: FieldDecl {{.*}} __cap_pinned 'const int *[32]'
// CHECK: FunctionDecl {{.*}} __gtap_state_machine_keep_const

#define __device__ __attribute__((device))
#define __GTAP_IS_BLOCK_MODE 1

namespace gtap::detail::block {
struct TaskContext {};
}
using TaskContext = gtap::detail::block::TaskContext;
struct uint3 { unsigned x, y, z; };
extern __device__ const uint3 threadIdx;
using TaskFn = void (*)(void *, int, TaskContext *);
__device__ void *__gtap_spawn_task(TaskContext *, int, int *, TaskFn, int);
__device__ void __gtap_finish_task(int, TaskContext *);
__device__ int __gtap_get_task_state(int);
__device__ bool __gtap_set_state_for_join_block(int, TaskContext *, int, int);

#pragma gtap function
__device__ int keep_const(const int *input) {
  const int saved = *input;
  const int *const pinned = input;
#pragma gtap taskwait
  return saved + *pinned;
}
