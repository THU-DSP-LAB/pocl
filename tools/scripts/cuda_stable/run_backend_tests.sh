#!/usr/bin/env bash

set -euo pipefail

readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly SOURCE_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"
readonly BUILD_DIR="${POCL_CUDA_BUILD_DIR:-$SOURCE_DIR/build_cuda_stable}"
readonly LOG_DIR="$BUILD_DIR/validation-logs"
readonly TEST_TIMEOUT_SECONDS=60

main() {
  if [[ ! -f "$BUILD_DIR/CTestTestfile.cmake" ]]; then
    echo "Missing configured test tree: $BUILD_DIR" >&2
    return 1
  fi

  mkdir -p "$LOG_DIR"
  env -u LD_LIBRARY_PATH POCL_DEVICES=cuda \
    ctest --test-dir "$BUILD_DIR" \
      -L cuda \
      --timeout "$TEST_TIMEOUT_SECONDS" \
      --output-on-failure \
    |& tee "$LOG_DIR/ctest-cuda.log"
}

main "$@"
