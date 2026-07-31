// RUN: %clang_cc1 -triple nvptx64-nvidia-cuda -fcuda-is-device -x cuda \
// RUN:   -std=c++17 -ast-dump %s | FileCheck %s
// CHECK: RecordDecl {{.*}} struct fib_block_task_data definition
// CHECK: FieldDecl {{.*}} n 'int[32]'
// CHECK: FieldDecl {{.*}} __cap_x 'int[32]'
// CHECK: FieldDecl {{.*}} __cap_y 'int[32]'
// CHECK: FieldDecl {{.*}} __gtap_spawning_thread 'int'
// CHECK: FunctionDecl {{.*}} __gtap_state_machine_fib_block
// CHECK: DeclRefExpr {{.*}} Function {{.*}} '__gtap_set_state_for_join_block'

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

#pragma gtap function
__device__ int fib_block(int n) {
  if (n < 2)
    return n;
  int x = 0;
  int y = 0;
  if (threadIdx.x == 0) {
#pragma gtap task
    x = fib_block(n - 1);
#pragma gtap task
    y = fib_block(n - 2);
  }
#pragma gtap taskwait
  return x + y;
}
