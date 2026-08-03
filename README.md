# Clang for GTaP

A fork of [LLVM/Clang](https://github.com/llvm/llvm-project) based on **LLVM 21.1.8** that adds **GTaP** support: pragma-based task parallelism for CUDA device code.

This compiler extends Clang so you can write GPU kernels using `#pragma gtap` directives (e.g. `#pragma gtap task`, `#pragma gtap taskwait`, `#pragma gtap entry`) and compile them to CUDA. It is intended to be used together with a GTaP runtime library.

## Features

- **Pragma-based API**: `#pragma gtap task`, `#pragma gtap taskwait`, `#pragma gtap entry`, `#pragma gtap function`, etc.
- **Frontend lowering to CUDA device code**: 
  GTaP constructs are transformed in the Clang frontend into low-level CUDA device code that cooperates with the GTaP runtime.
  In particular, fork-join constructs (e.g. `task` / `taskwait`) are lowered into a state-machine style representation that enables suspension and resumption on GPUs.
- **CUDA integration**: Compiles to `NVPTX`; requires CUDA Toolkit and a GTaP runtime.
- **Based on LLVM/Clang 21.1.8**: Built on the official LLVM monorepo with minimal, targeted changes to the Clang frontend (parsing, AST, sema).

## Project layout (GTaP-related)

- **`llvm/`** — LLVM core (unmodified from upstream layout).
- **`clang/`** — Clang with GTaP additions, including:
  - Parsing: `ParseGTaP.cpp`, pragma handling
  - AST: `StmtGTaP.cpp`, `StmtGTaP.h`, `GTaPTaskInfo.h`
  - Semantics: `SemaGTaP.cpp`, `SemaGTaP.h`, `GTaPKinds.h` / `GTaPKinds.def`
  - Serialization: GTaP statement kinds in AST reader/writer

## Requirements

- CMake 3.x
- Ninja (recommended) or another CMake generator
- C/C++ toolchain (Clang or GCC)
- [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads)

We have verified build and basic functionality on a single GH200 node of the [Miyabi-G](https://www.cc.u-tokyo.ac.jp/en/supercomputer/miyabi/service/) supercomputer (1× NVIDIA GH200; Clang 21.1.8, CUDA Toolkit 12.9, Linux kernel `5.14.0-427.13.1.el9_4.aarch64`).

## Building

Configure and build from the repository root (same layout as [llvm-project](https://github.com/llvm/llvm-project)):

```bash
cd /path/to/clang-gtap

cmake -G Ninja \
  -S llvm -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=clang \
  -DCMAKE_CXX_COMPILER=clang++ \
  -DLLVM_ENABLE_PROJECTS="clang;clang-tools-extra" \
  -DLLVM_TARGETS_TO_BUILD="AArch64;NVPTX" \
  -DCLANG_ENABLE_CUDA=ON \
  -DLLVM_ENABLE_RUNTIMES="openmp" \
  -DCUDA_TOOLKIT_ROOT_DIR=/path/to/cuda
```

Adjust `LLVM_TARGETS_TO_BUILD` for your host (e.g. add `X86` for `x86_64`) and set `CUDA_TOOLKIT_ROOT_DIR` to your CUDA installation. Then:

```bash
ninja -C build clang
```

The resulting `clang` (and related binaries) will be in `build/bin/`. Use `build/bin/clang` to compile GTaP programs.

## License and attribution

This project is a **derivative of the LLVM Project**. The entire codebase is distributed under the **Apache License v2.0 with LLVM Exceptions**.

- Full license text: [`LICENSE.TXT`](LICENSE.TXT) (repository root) and [`llvm/LICENSE.TXT`](llvm/LICENSE.TXT).

## Links

- [LLVM Project](https://llvm.org/)
- [Clang](https://clang.llvm.org/)
- [Apache License, Version 2.0](https://www.apache.org/licenses/LICENSE-2.0)
- [LLVM Exceptions to the Apache 2.0 License](https://llvm.org/docs/DeveloperPolicy.html#license)
