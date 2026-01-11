#!/bin/bash

set -euo pipefail

PROJ_DIR="/work/gc64/c64099"
CLANG_BIN="${PROJ_DIR}/clang-gtap/build/bin/clang"
SRC_DIR="${PROJ_DIR}/clang-gtap/example/thread"
OUT_DIR="${SRC_DIR}/out"

mkdir -p "${OUT_DIR}"

# Device-side AST dump, filtered to declarations containing "test"
"${CLANG_BIN}" \
  -O0 \
  -x cuda \
  -Xclang -fcuda-is-device \
  -fsyntax-only \
  -Xclang -ast-dump \
  -Xclang -ast-dump-filter=for_task \
  -I"${PROJ_DIR}/gpu-task-runtime/thread/include" \
  -I"${PROJ_DIR}/gpu-task-runtime/common" \
  -D__CUDACC__ \
  "${SRC_DIR}/test.cu" \
  > "${OUT_DIR}/test_ast_device.txt" 2>&1

echo "Filtered AST (test*) dumped to ${OUT_DIR}/test_ast_device.txt"
