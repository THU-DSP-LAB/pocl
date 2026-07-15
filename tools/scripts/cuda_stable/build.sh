#!/usr/bin/env bash

set -euo pipefail

readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly SOURCE_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"
readonly BUILD_DIR="${POCL_CUDA_BUILD_DIR:-$SOURCE_DIR/build_cuda_stable}"
readonly INSTALL_DIR="${POCL_CUDA_INSTALL_DIR:-$BUILD_DIR/install}"
readonly LOG_DIR="$BUILD_DIR/validation-logs/build"
readonly JOBS="${POCL_BUILD_JOBS:-$(nproc)}"
readonly LLVM_CONFIG="${LLVM_CONFIG:-/usr/bin/llvm-config-18}"

require_command() {
  local command_name=$1
  if ! command -v "$command_name" >/dev/null; then
    echo "Missing required command: $command_name" >&2
    return 1
  fi
}

verify_toolchain() {
  require_command cmake
  require_command ninja
  require_command nvidia-smi
  if [[ ! -x "$LLVM_CONFIG" ]]; then
    echo "Missing LLVM config executable: $LLVM_CONFIG" >&2
    return 1
  fi

  local llvm_version
  llvm_version="$($LLVM_CONFIG --version)"
  if [[ "$llvm_version" != "18.1.3" ]]; then
    echo "Expected LLVM 18.1.3, found $llvm_version" >&2
    return 1
  fi

  local grammar=/usr/include/spirv/unified1/spirv.core.grammar.json
  if [[ ! -f "$grammar" ]]; then
    echo "Missing SPIR-V headers: $grammar" >&2
    return 1
  fi
}

configure() {
  cmake -S "$SOURCE_DIR" -B "$BUILD_DIR" --fresh -G Ninja \
    -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    -DCMAKE_INSTALL_PREFIX="$INSTALL_DIR" \
    -DWITH_LLVM_CONFIG="$LLVM_CONFIG" \
    -DENABLE_CONFORMANCE=ON \
    -DENABLE_CUDA=ON \
    -DENABLE_EXAMPLES=ON \
    -DENABLE_HOST_CPU_DEVICES=OFF \
    -DENABLE_HWLOC=OFF \
    -DENABLE_ICD=OFF \
    -DENABLE_SPIRV=OFF \
    -DENABLE_TESTS=ON \
    -DENABLE_TESTSUITES=conformance \
    -DCTS_SPIRV_INCLUDE_DIR=/usr \
    |& tee "$LOG_DIR/configure.log"
}

build_and_install() {
  cmake --build "$BUILD_DIR" --clean-first --parallel "$JOBS" \
    |& tee "$LOG_DIR/build.log"
  cmake --install "$BUILD_DIR" |& tee "$LOG_DIR/install.log"
  local opencl_soname
  opencl_soname="$(readlink "$INSTALL_DIR/lib/libOpenCL.so.2")"
  cmake -E create_symlink "$opencl_soname" "$INSTALL_DIR/lib/libOpenCL.so.1"
  printf 'libOpenCL.so.1 -> %s\n' "$opencl_soname" \
    | tee -a "$LOG_DIR/install.log"
  cmake --build "$BUILD_DIR" --target conformance --parallel "$JOBS" \
    |& tee "$LOG_DIR/build-cts.log"
}

main() {
  verify_toolchain
  mkdir -p "$LOG_DIR"
  configure
  build_and_install
  printf 'Build directory: %s\nInstall directory: %s\n' \
    "$BUILD_DIR" "$INSTALL_DIR"
}

main "$@"
