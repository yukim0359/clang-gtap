#!/bin/bash

set -euo pipefail

PROJ_DIR="/work/gc64/c64099"
CLANG_BIN="${PROJ_DIR}/clang-gtap/build/bin/clang"
SRC_DIR="${PROJ_DIR}/clang-gtap/example/thread"
OUT_DIR="${SRC_DIR}/out"

mkdir -p "${OUT_DIR}"

# Device-side AST dump, filtered to declarations containing "mergesort"
"${CLANG_BIN}" \
  -O0 \
  -x cuda \
  -Xclang -fcuda-is-device \
  -fsyntax-only \
  -Xclang -ast-dump \
  -Xclang -ast-dump-filter=mergesort \
  -I"${PROJ_DIR}/gpu-task-runtime/thread/include" \
  -I"${PROJ_DIR}/gpu-task-runtime/common" \
  -D__CUDACC__ \
  "${SRC_DIR}/mergesort_ideal.cu" \
  > "${OUT_DIR}/mergesort_ast_device.txt" 2>&1

echo "Filtered AST (mergesort*) dumped to ${OUT_DIR}/mergesort_ast_device.txt"
