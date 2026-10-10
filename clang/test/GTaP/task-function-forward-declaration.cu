// RUN: %clang_cc1 -triple nvptx64-nvidia-cuda -fcuda-is-device -x cuda \
// RUN:   -std=c++17 -ast-dump %s | FileCheck %s
// CHECK: RecordDecl {{.*}} struct definition_mutates_task_data definition
// CHECK: FieldDecl {{.*}} x 'int[32]'
// CHECK: FunctionDecl {{.*}} __gtap_state_machine_definition_mutates
// CHECK: RecordDecl {{.*}} struct definition_read_only_task_data definition
// CHECK: FieldDecl {{.*}} x 'int'
// CHECK: FunctionDecl {{.*}} __gtap_state_machine_definition_read_only
// CHECK: CompoundStmt

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
__device__ bool __gtap_prepare_for_join_block(int, TaskContext *, int, int);

#pragma gtap function
__device__ int definition_mutates(const int);

__device__ int definition_mutates(int x) {
  ++x;
  return x;
}

#pragma gtap function
__device__ int definition_read_only(int);

__device__ int definition_read_only(const int x) { return x + 1; }
