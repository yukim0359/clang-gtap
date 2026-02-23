#!/bin/bash

set -euo pipefail

PROJ_DIR="/work/gc64/c64099/gtap"
CLANG_BIN="${PROJ_DIR}/clang-gtap/build/bin/clang"
SRC_DIR="${PROJ_DIR}/clang-gtap/example/thread"
OUT_DIR="${SRC_DIR}/out"

mkdir -p "${OUT_DIR}"

# Device-side AST dump, filtered to declarations containing "cilksort"
"${CLANG_BIN}" \
  -O0 \
  -x cuda \
  -Xclang -fcuda-is-device \
  -fsyntax-only \
  -Xclang -ast-dump \
  -Xclang -ast-dump-filter=_task \
  -I"${PROJ_DIR}/runtime" \
  -D__CUDACC__ \
  "${SRC_DIR}/cilksort_ideal.cu" \
  > "${OUT_DIR}/cilksort_ideal_ast_device.txt" 2>&1

echo "Filtered AST (cilksort*) dumped to ${OUT_DIR}/cilksort_ideal_ast_device.txt"
