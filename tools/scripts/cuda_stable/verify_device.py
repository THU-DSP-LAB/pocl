#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from pathlib import Path

from opencl_probe import DeviceSnapshot, snapshot

EXPECTED_EXTENSIONS = frozenset(
    {
        "cl_khr_byte_addressable_store",
        "cl_khr_global_int32_base_atomics",
        "cl_khr_global_int32_extended_atomics",
        "cl_khr_local_int32_base_atomics",
        "cl_khr_local_int32_extended_atomics",
        "cl_khr_fp64",
        "cl_nv_device_attribute_query",
    }
)
EXPECTED_FEATURES = frozenset({"__opencl_c_fp64", "__opencl_c_int64"})
EXPECTED_CONFORMANCE_VERSION = "v2026-07-14-00"
CL_FP_DENORM = 1 << 0
CL_FP_INF_NAN = 1 << 1
CL_FP_ROUND_TO_NEAREST = 1 << 2
CL_FP_ROUND_TO_ZERO = 1 << 3
CL_FP_ROUND_TO_INF = 1 << 4
CL_FP_FMA = 1 << 5
CUDA_SINGLE_FP_CONFIG = (
    CL_FP_DENORM
    | CL_FP_INF_NAN
    | CL_FP_ROUND_TO_NEAREST
    | CL_FP_ROUND_TO_ZERO
    | CL_FP_ROUND_TO_INF
    | CL_FP_FMA
)
CUDA_DOUBLE_FP_CONFIG = CUDA_SINGLE_FP_CONFIG
EXPECTED_SCALARS = {
    "image_support": 0,
    "svm_capabilities": 0,
    "queue_properties": 1 << 1,
    "max_parameter_size": 4352 - 4 * 4,
    "max_num_sub_groups": 0,
    "subgroup_forward_progress": 0,
    "atomic_memory_capabilities": (1 << 0) | (1 << 4),
    "atomic_fence_capabilities": (1 << 0) | (1 << 1) | (1 << 4),
    "non_uniform_support": 0,
    "generic_address_support": 0,
    "pipe_support": 0,
    "single_fp_config": CUDA_SINGLE_FP_CONFIG,
    "double_fp_config": CUDA_DOUBLE_FP_CONFIG,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit the fixed PoCL CUDA device contract")
    parser.add_argument("--library", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def validate_strings(device: DeviceSnapshot) -> list[str]:
    failures: list[str] = []
    expected_strings = {
        "name": "NVIDIA GeForce RTX 4090",
        "opencl_c_version": "OpenCL C 1.2 PoCL",
        "il_version": "",
        "latest_conformance": EXPECTED_CONFORMANCE_VERSION,
    }
    for field, expected in expected_strings.items():
        actual = getattr(device, field)
        if actual != expected:
            failures.append(f"{field}: expected {expected!r}, found {actual!r}")
    if "OpenCL 3.0" not in device.device_version or "CUDA-sm_89" not in device.device_version:
        failures.append(f"device_version does not identify OpenCL 3.0 CUDA-sm_89: {device.device_version}")
    return failures


def validate_capabilities(device: DeviceSnapshot) -> list[str]:
    failures: list[str] = []
    extensions = frozenset(device.extensions)
    versioned = frozenset(device.versioned_extensions)
    features = frozenset(device.features)
    if extensions != EXPECTED_EXTENSIONS:
        failures.append(f"extensions: expected {sorted(EXPECTED_EXTENSIONS)}, found {sorted(extensions)}")
    if versioned != extensions:
        failures.append(f"versioned extensions differ from extension string: {sorted(versioned)}")
    if features != EXPECTED_FEATURES:
        failures.append(f"features: expected {sorted(EXPECTED_FEATURES)}, found {sorted(features)}")
    if device.half_fp_config != 0 or device.half_fp_query_error != 0:
        failures.append(
            "CL_DEVICE_HALF_FP_CONFIG must return zero successfully when cl_khr_fp16 "
            f"is absent; value={device.half_fp_config}, error={device.half_fp_query_error}"
        )
    for field, expected in EXPECTED_SCALARS.items():
        actual = getattr(device, field)
        if actual != expected:
            failures.append(f"{field}: expected {expected}, found {actual}")
    return failures


def main() -> int:
    args = parse_args()
    os.environ["POCL_DEVICES"] = "cuda"
    os.environ.pop("POCL_CUDA_GPU_ARCH", None)
    device = snapshot(args.library)
    report = asdict(device)
    report["failures"] = validate_strings(device) + validate_capabilities(device)
    serialized = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized, encoding="utf-8")
    print(serialized, end="")
    return 1 if report["failures"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
