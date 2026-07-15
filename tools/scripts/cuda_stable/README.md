# PoCL CUDA fixed-baseline validation

This directory contains the reproducible build and validation entry points for
the RTX 4090 baseline. The supported environment is intentionally singular:
NVIDIA RTX 4090, driver 590.48.01, LLVM 18.1.3, CUDA target `sm_89`, OpenCL 3.0,
and OpenCL C 1.2.

## Dependencies

On Ubuntu 24.04, install the PoCL build dependencies plus the pinned LLVM and
CTS inputs:

```sh
sudo apt-get update
sudo apt-get install llvm-18-dev libclang-18-dev libclang-cpp18-dev \
  spirv-headers spirv-tools ninja-build libgl-dev libglu1-mesa-dev \
  freeglut3-dev libglew-dev libx11-dev
```

The SPIR-V headers are a CTS build dependency. The CUDA device does not expose
SPIR or SPIR-V IL support in this baseline.

## Clean build and install

From a clean checkout:

```sh
tools/scripts/cuda_stable/build.sh
```

The default build and install roots are `build_cuda_stable/` and
`build_cuda_stable/install/`. Override them with `POCL_CUDA_BUILD_DIR` and
`POCL_CUDA_INSTALL_DIR`. Configure, build, install, and CTS build logs are kept
under `build_cuda_stable/validation-logs/build/`.

## Device contract

```sh
POCL_DEVICES=cuda \
  tools/scripts/cuda_stable/verify_device.py \
  --library build_cuda_stable/install/lib/libOpenCL.so
```

The audit fails on any undeclared architecture, extension-list mismatch, or
optional capability outside the fixed baseline. It loads the requested PoCL
library by absolute path.

## CUDA backend tests

```sh
tools/scripts/cuda_stable/run_backend_tests.sh
```

The backend tests use the build-tree library and CUDA plugin, with a hard
60-second timeout per test. The script deliberately removes `LD_LIBRARY_PATH`
so an installed library cannot be mixed with the build-tree device plugin.

## CTS quick

Run only while the GPU has no compute clients:

```sh
tools/scripts/cuda_stable/run_cts_quick.py
```

Each CSV entry runs in its own process group. The default per-suite watchdog is
900 seconds because several applicable quick suites legitimately exceed one
minute; pass `--timeout 0` to disable the watchdog explicitly. A timeout sends
`SIGTERM` to the whole group and then `SIGKILL` if termination has not completed
after five seconds. Raw stdout, raw stderr, exit status, signal, environment,
loader path, and an incrementally updated JSON summary are retained below
`build_cuda_stable/validation-logs/cts-quick-<UTC timestamp>/`.

The runner rejects a busy GPU and rejects every CTS binary whose dynamic
`libOpenCL` resolution is not the installed PoCL library. Test selection
with `--filter` is for root-cause reproduction only and is not an acceptance
run.

## CTS expanded

Goal 2 adds a version-controlled manifest for official CTS modes omitted or
reduced by the quick CSV:

```sh
tools/scripts/cuda_stable/run_cts_quick.py --suite expanded --timeout 1800
```

The expanded tier runs full thread-dimension, multiple-context, vector-layout,
integer, and half suites, wimpy conversions, and selected non-wimpy math
functions. Long conversions, integer, `rootn`, and `remquo` modes are split
into explicit, non-overlapping CTS ranges or registered test/type/vector modes
so each process remains within the watchdog without reducing coverage. It uses
the same loader, device-contract, process-group, raw-log, and GPU-residue
checks as quick. An
arbitrary version-controlled manifest can be selected explicitly with
`--csv PATH`. For a warm-cache repeat with a separate immutable result
directory, pass the first run's cache explicitly with `--cache-dir PATH`.
