// RUN: %clang_cc1 -triple nvptx64-nvidia-cuda -fcuda-is-device -x cuda \
// RUN:   -std=c++17 -ast-dump %s | FileCheck %s
// CHECK: RecordDecl {{.*}} struct uniform_parameters_task_data definition
// CHECK: FieldDecl {{.*}} input 'const Block *[32]'
// CHECK: FieldDecl {{.*}} uniform_input 'const Block *'
// CHECK: FieldDecl {{.*}} uniform_x 'int'
// CHECK: FieldDecl {{.*}} mutable_x 'int[32]'
// CHECK: FieldDecl {{.*}} __gtap_spawning_thread 'int'

#define __device__ __attribute__((device))
#define GTAP_BLOCK_SIZE 32
#define __GTAP_WORKER_IS_BLOCK 1

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

#pragma gtap function
__device__ int uniform_parameters(const Block *input,
                                  const Block *const uniform_input,
                                  const int uniform_x, int mutable_x) {
  input += threadIdx.x;
  mutable_x += threadIdx.x;
  return input->value + uniform_input->value + uniform_x + mutable_x;
}
