#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PROJ_DIR="/work/gc64/c64099"

: "${CLANG_BIN:=${ROOT_DIR}/build/bin/clang}"
: "${CUDA_PATH:=/work/opt/local/aarch64/cores/nvidia/25.9/Linux_aarch64/25.9/cuda}"
: "${CUDA_ARCH:=sm_90}"

SRC="${SCRIPT_DIR}/test.cu"
OUT_DIR="${SCRIPT_DIR}/out"
OUT_EXEC="${OUT_DIR}/test"

# gpu-task-runtime includes
GT_INC1="${PROJ_DIR}/gpu-task-runtime/thread/include"
GT_INC2="${PROJ_DIR}/gpu-task-runtime/common"

# launch/queue defaults (can be overridden via env)
: "${NUM_BLOCKS:=1000}"
: "${THREADS_PER_BLK:=128}"
: "${MAX_TASKS_PER_WARP:=20000}"
: "${MAX_CHILD_TASKS:=40}"
: "${TASK_KIND:=1}"

echo "============================================"
echo "  GTaP Test Script"
echo "============================================"
echo "[info] CLANG_BIN=${CLANG_BIN}"
echo "[info] CUDA_PATH=${CUDA_PATH}"
echo "[info] CUDA_ARCH=${CUDA_ARCH}"
echo "[info] SRC=${SRC}"
echo "[info] OUT_DIR=${OUT_DIR}"
echo "[info] CFG: NUM_BLOCKS=${NUM_BLOCKS} THREADS_PER_BLK=${THREADS_PER_BLK}"
echo "[info]      MAX_TASKS_PER_WARP=${MAX_TASKS_PER_WARP} MAX_CHILD_TASKS=${MAX_CHILD_TASKS}"
echo "[info]      TASK_KIND=${TASK_KIND}"
echo ""

mkdir -p "${OUT_DIR}"

# Frontend-only (device) with time-trace: AST/Sema までで止まるかを確認
echo "[step 1/4] Device front-end only (syntax check)..."
"${CLANG_BIN}" -O0 -x cuda -fsyntax-only -v \
  -I"${GT_INC1}" -I"${GT_INC2}" \
  -DNUM_BLOCKS="${NUM_BLOCKS}" -DTHREADS_PER_BLK="${THREADS_PER_BLK}" \
  -DMAX_TASKS_PER_WARP="${MAX_TASKS_PER_WARP}" -DMAX_CHILD_TASKS="${MAX_CHILD_TASKS}" \
  -DTASK_KIND="${TASK_KIND}" \
  -DGTAP_TERMINATE_ON_FIRST_TASK_FINISH \
  -D__CUDACC__ \
  "${SRC}" > "${OUT_DIR}/device_syntax_only.log" 2>&1 || {
  echo "[error] Device syntax-only check failed"
  exit 1
}
echo "[ok] Device syntax-only done (see ${OUT_DIR}/device_syntax_only.log)"

# Frontend-only (host) with time-trace
echo "[step 2/4] Host front-end only (syntax check)..."
"${CLANG_BIN}" -O0 -x cuda -fsyntax-only -v \
  -I"${GT_INC1}" -I"${GT_INC2}" \
  -DNUM_BLOCKS="${NUM_BLOCKS}" -DTHREADS_PER_BLK="${THREADS_PER_BLK}" \
  -DMAX_TASKS_PER_WARP="${MAX_TASKS_PER_WARP}" -DMAX_CHILD_TASKS="${MAX_CHILD_TASKS}" \
  -DTASK_KIND="${TASK_KIND}" \
  -DGTAP_TERMINATE_ON_FIRST_TASK_FINISH \
  -D__CUDACC__ \
  "${SRC}" > "${OUT_DIR}/host_syntax_only.log" 2>&1 || {
  echo "[warn] Host syntax-only check failed"
}
echo "[ok] Host syntax-only done (see ${OUT_DIR}/host_syntax_only.log)"

# Compile directly to executable using clang (handles both host and device)
echo "[step 3/4] Compiling to executable using clang..."
"${CLANG_BIN}" -O3 -x cuda \
  --cuda-path="${CUDA_PATH}" \
  --cuda-gpu-arch="${CUDA_ARCH}" \
  -Wall -Wextra \
  -v \
  -I"${GT_INC1}" -I"${GT_INC2}" \
  -DNUM_BLOCKS="${NUM_BLOCKS}" -DTHREADS_PER_BLK="${THREADS_PER_BLK}" \
  -DMAX_TASKS_PER_WARP="${MAX_TASKS_PER_WARP}" -DMAX_CHILD_TASKS="${MAX_CHILD_TASKS}" \
  -DTASK_KIND="${TASK_KIND}" \
  -DGTAP_TERMINATE_ON_FIRST_TASK_FINISH \
  -L"${CUDA_PATH}/lib64" -lcudart \
  "${SRC}" -o "${OUT_EXEC}" 2>&1 | tee "${OUT_DIR}/test_compile.log"

if [ ! -f "${OUT_EXEC}" ]; then
  echo "[ERROR] Compilation failed"
  exit 1
fi
echo "[OK] Executable generated: ${OUT_EXEC}"
echo ""

# Step 4: Run the program
echo "[step 4/4] Running the program..."
echo "============================================"
export LD_LIBRARY_PATH="${CUDA_PATH}/lib64:${ROOT_DIR}/build/lib:${LD_LIBRARY_PATH:-}"
"${OUT_EXEC}"
EXIT_CODE=$?
echo "============================================"
echo ""

if [ ${EXIT_CODE} -eq 0 ]; then
  echo "[SUCCESS] Program executed successfully!"
else
  echo "[ERROR] Program exited with code ${EXIT_CODE}"
  exit ${EXIT_CODE}
fi

echo ""
echo "============================================"
echo "  Test completed!"
echo "============================================"
echo ""
echo "Generated files:"
echo "  - Executable: ${OUT_EXEC}"
echo "  - Compile log: ${OUT_DIR}/test_compile.log"
echo "  - Syntax logs: ${OUT_DIR}/*_syntax_only.log"
