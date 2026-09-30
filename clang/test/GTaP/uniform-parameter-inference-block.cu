// RUN: %clang_cc1 -triple nvptx64-nvidia-cuda -fcuda-is-device -x cuda \
// RUN:   -std=c++17 -ast-dump %s | FileCheck %s
// Verify that block-task parameters are stored uniformly unless their own
// value is mutated or may escape.
// CHECK: RecordDecl {{.*}} struct inferred_parameters_task_data definition
// CHECK: FieldDecl {{.*}} input 'const Block *'
// CHECK: FieldDecl {{.*}} output 'Block *'
// CHECK: FieldDecl {{.*}} read_only 'int'
// CHECK: FieldDecl {{.*}} top_const_pointer 'const Block *'
// CHECK: FieldDecl {{.*}} __gtap_spawning_thread 'int'
// CHECK: FieldDecl {{.*}} __gtap_result 'int'
// CHECK: FieldDecl {{.*}} __gtap_result_dst 'int *'
// CHECK: RecordDecl {{.*}} struct inferred_parameters_task_lane_storage definition
// CHECK: FieldDecl {{.*}} assigned 'int[32]'
// CHECK: FieldDecl {{.*}} incremented 'int[32]'
// CHECK: FieldDecl {{.*}} compound 'int[32]'
// CHECK: FieldDecl {{.*}} aggregate 'Block[32]'
// CHECK: FieldDecl {{.*}} by_ref 'int[32]'
// CHECK: FieldDecl {{.*}} address_taken 'int[32]'

#define __device__ __attribute__((device))
#define __GTAP_IS_BLOCK_MODE 1

struct TaskContext {};
struct uint3 { unsigned x, y, z; };
extern __device__ const uint3 threadIdx;
constexpr unsigned long long __gtap_max_task_size = ~0ULL;
using TaskFn = void (*)(void *, int, TaskContext *);
__device__ void *__gtap_spawn_task(TaskContext *, int, int *, TaskFn, int);
__device__ void __gtap_finish_task(int, TaskContext *);
__device__ int __gtap_get_task_state(int);
__device__ bool __gtap_set_state_for_join_block(int, TaskContext *, int, int);

struct Block {
  int value;
};

__device__ void mutate(int &);
__device__ void observe_address(int *);

#pragma gtap function
__device__ int inferred_parameters(const Block *input, Block *output,
                                   int read_only, int assigned,
                                   int incremented, int compound,
                                   Block aggregate, int by_ref,
                                   int address_taken,
                                   const Block *const top_const_pointer) {
  output->value = read_only;
  assigned = read_only;
  ++incremented;
  compound += read_only;
  aggregate.value++;
  mutate(by_ref);
  observe_address(&address_taken);
  return input->value + output->value + read_only + assigned + incremented +
         compound + aggregate.value + by_ref + address_taken +
         top_const_pointer->value;
}
