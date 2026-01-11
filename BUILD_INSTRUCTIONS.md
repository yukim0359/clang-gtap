```bash
cd ~/clang-gtap
cmake -G Ninja \
  -S llvm -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=clang \
  -DCMAKE_CXX_COMPILER=clang++ \
  -DLLVM_ENABLE_PROJECTS="clang;clang-tools-extra" \
  -DLLVM_TARGETS_TO_BUILD="AArch64;NVPTX" \
  -DCLANG_ENABLE_CUDA=ON \
  -DCUDA_TOOLKIT_ROOT_DIR=/work/opt/local/aarch64/cores/nvidia/25.9/Linux_aarch64/25.9/cuda
```
