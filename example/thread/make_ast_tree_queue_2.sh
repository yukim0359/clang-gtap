#!/bin/bash

set -euo pipefail

PROJ_DIR="/work/gc64/c64099/gtap"
CLANG_BIN="${PROJ_DIR}/clang-gtap/build/bin/clang++"
SRC_DIR="${PROJ_DIR}/clang-gtap/example/thread"
OUT_DIR="${SRC_DIR}/out"

mkdir -p "${OUT_DIR}"

# Device-side AST dump, filtered to declarations containing "tree_work"
"${CLANG_BIN}" \
  -O3 \
  -x cuda \
  -Xclang -fcuda-is-device \
  -fsyntax-only \
  -Xclang -ast-dump \
  -Xclang -ast-dump-filter=tree_work \
  -I"${PROJ_DIR}/runtime" \
  -D__CUDACC__ \
  "${SRC_DIR}/tree_queue_2.cu" \
  > "${OUT_DIR}/tree_queue_2_ast_device.txt" 2>&1

echo "Filtered AST (tree_work*) dumped to ${OUT_DIR}/tree_queue_2_ast_device.txt"
