// RUN: %clang_cc1 -triple nvptx64-nvidia-cuda -fcuda-is-device -x cuda \
// RUN:   -std=c++17 -ast-dump %s | FileCheck %s
// CHECK: RecordDecl {{.*}} struct captured_arrays_task_data definition
// CHECK: FieldDecl {{.*}} input 'const Block *'
// CHECK: FieldDecl {{.*}} __cap_pending_a 'const Block *[32]'
// CHECK: FieldDecl {{.*}} __cap_pending_y 'int[32]'
// CHECK: FieldDecl {{.*}} __cap_pending_blocks 'Block[4]'
// CHECK: FieldDecl {{.*}} __cap_constants 'const int[2]'
// CHECK: FunctionDecl {{.*}} __gtap_state_machine_captured_arrays
// CHECK: CompoundStmt

#define __device__ __attribute__((device))

struct TaskContext {};
constexpr unsigned long long __gtap_max_task_size = ~0ULL;
using TaskFn = void (*)(void *, int, TaskContext *);
__device__ void *__gtap_spawn_task(TaskContext *, int, int *, TaskFn, int);
__device__ void __gtap_finish_task(int, TaskContext *);
__device__ int __gtap_get_task_state(int);
__device__ bool __gtap_set_state_for_join(int, int, int, int);

struct Block {
  int value;
};

#pragma gtap function
__device__ int leaf(int x) { return x + 1; }

#pragma gtap function
__device__ int captured_arrays(const Block *input) {
  const Block *pending_a[32] = {};
  int pending_y[32] = {};
  Block pending_blocks[4] = {};
  const int constants[2] = {3, 4};

  pending_a[0] = input;
  pending_y[1] = input->value;
  pending_blocks[2].value = 5;

  int child;
#pragma gtap task
  child = leaf(input->value);
#pragma gtap taskwait
  return pending_a[0]->value + pending_y[1] +
         pending_blocks[2].value + constants[1] + child;
}
