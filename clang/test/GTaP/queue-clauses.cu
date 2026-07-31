// RUN: %clang_cc1 -triple nvptx64-nvidia-cuda -fcuda-is-device -x cuda \
// RUN:   -std=c++17 -ast-dump %s | FileCheck %s
// CHECK: RecordDecl {{.*}} struct queued_task_data definition
// CHECK: FieldDecl {{.*}} spawn_queue 'int'
// CHECK: FieldDecl {{.*}} resume_queue 'int'
// CHECK: FunctionDecl {{.*}} __gtap_state_machine_queued
// CHECK: DeclRefExpr {{.*}} Function {{.*}} '__gtap_spawn_task'
// CHECK: MemberExpr {{.*}}spawn_queue
// CHECK: DeclRefExpr {{.*}} Function {{.*}} '__gtap_set_state_for_join'
// CHECK: MemberExpr {{.*}}resume_queue

#define __device__ __attribute__((device))

struct TaskContext {};
constexpr unsigned long long __gtap_max_task_size = ~0ULL;
using TaskFn = void (*)(void *, int, TaskContext *);
__device__ void *__gtap_spawn_task(TaskContext *, int, int *, TaskFn, int);
__device__ void __gtap_finish_task(int, TaskContext *);
__device__ int __gtap_get_task_state(int);
__device__ bool __gtap_set_state_for_join(int, int, int, int);

#pragma gtap function
__device__ int leaf(int x) { return x; }

#pragma gtap function
__device__ int queued(int x, int spawn_queue, int resume_queue) {
  int result;
#pragma gtap task queue(spawn_queue)
  result = leaf(x);
#pragma gtap taskwait queue(resume_queue)
  return result;
}
