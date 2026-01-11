#!/bin/bash

set -euo pipefail

PROJ_DIR="/work/gc64/c64099"
CLANG_BIN="${PROJ_DIR}/clang-gtap/build/bin/clang"
SRC_DIR="${PROJ_DIR}/clang-gtap/example/thread"
OUT_DIR="${SRC_DIR}/out"

mkdir -p "${OUT_DIR}"

# Device-side LLVM IR dump for nq.cu
echo "Generating LLVM IR for nq.cu (device side)..."
"${CLANG_BIN}" \
  -O3 \
  -x cuda \
  -Xclang -fcuda-is-device \
  --cuda-device-only \
  -emit-llvm \
  -S \
  -I"${PROJ_DIR}/gpu-task-runtime/thread/include" \
  -I"${PROJ_DIR}/gpu-task-runtime/common" \
  -DNUM_BLOCKS=2000 \
  -DTHREADS_PER_BLK=64 \
  -DMAX_TASKS_PER_WARP=80000 \
  -DMAX_CHILD_TASKS=16 \
  -DTASK_KIND=1 \
  -D__CUDACC__ \
  "${SRC_DIR}/nq.cu" \
  -o "${OUT_DIR}/nq_device.ll"

echo "Device-side LLVM IR dumped to ${OUT_DIR}/nq_device.ll"

# Host-side LLVM IR dump for nq.cu (if needed)
echo "Generating LLVM IR for nq.cu (host side)..."
"${CLANG_BIN}" \
  -O3 \
  -x cuda \
  --cuda-host-only \
  -emit-llvm \
  -S \
  -I"${PROJ_DIR}/gpu-task-runtime/thread/include" \
  -I"${PROJ_DIR}/gpu-task-runtime/common" \
  -DNUM_BLOCKS=2000 \
  -DTHREADS_PER_BLK=64 \
  -DMAX_TASKS_PER_WARP=80000 \
  -DMAX_CHILD_TASKS=16 \
  -DTASK_KIND=1 \
  -D__CUDACC__ \
  "${SRC_DIR}/nq.cu" \
  -o "${OUT_DIR}/nq_host.ll"

echo "Host-side LLVM IR dumped to ${OUT_DIR}/nq_host.ll"
