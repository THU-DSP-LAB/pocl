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
  spirv-headers spirv-tools ninja-build libxml2-dev
```

The SPIR-V headers are a CTS build dependency. OpenCL CTS itself is provided by
the top-level `third_party/OpenCL-CTS` submodule and is never fetched by CMake.
The CUDA device does not expose SPIR, SPIR-V IL, or OpenGL interoperability in
this baseline, so the corresponding CTS modes are disabled.

## Clean build and install

From a clean checkout:

```sh
tools/scripts/cuda_stable/build.sh
```

The default build tree is `<repository>/build/pocl/`, and every component is
installed into the shared `<repository>/install/` prefix. Override them with
`POCL_CUDA_BUILD_DIR` and `INSTALL_DIR`. Configure, build, install, and CTS
build logs are kept under `<repository>/build/pocl/validation-logs/build/`.

## Device contract

```sh
POCL_DEVICES=cuda \
  tools/scripts/cuda_stable/verify_device.py \
  --library ../install/lib/libOpenCL.so
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
`<repository>/build/pocl/validation-logs/cts-quick-<UTC timestamp>/`.

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

## CTS Goal 3 complete Conversions and Math

Audit the fixed CTS function lists, device capability exclusions, and exact
manifest partition before starting the long run:

```sh
tools/scripts/cuda_stable/cts_goal3_manifest.py \
  --cts-dir ../build/pocl/examples/conformance/src/conformance-build/test_conformance \
  --install-dir ../install \
  --device-contract PATH/TO/device-contract.json
```

The Goal 3 manifest contains 105 continuous Conversions ranges covering test
numbers 1 through 1045 exactly once, plus FP32 and FP64 shards for all 101
applicable Math functions. Math shards retain the default scalar/v2/v3/v4/v8/v16
coverage. `divide_cr` and `sqrt_cr` are recorded as capability N/A because the
fixed device contract does not advertise correctly-rounded divide/sqrt.

Run cold and warm acceptance passes with a watchdog sized for indivisible
brute-force functions:

```sh
tools/scripts/cuda_stable/run_cts_quick.py \
  --suite goal3 --timeout 43200 --output-dir PATH/TO/goal3-cold
tools/scripts/cuda_stable/run_cts_quick.py \
  --suite goal3 --timeout 43200 --output-dir PATH/TO/goal3-warm \
  --cache-dir PATH/TO/goal3-cold/kernel-cache
```

The runner records applicable pass counts and per-test N/A names/reasons. A
suite whose selected registered tests are all unsupported for one documented
capability is classified as `skip`; mixed applicable/N/A suites remain `pass`.
Timeouts, signals, nonzero exit codes, and failure markers take precedence.
