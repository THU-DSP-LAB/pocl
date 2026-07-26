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
        "cl_khr_int64_base_atomics",
        "cl_khr_int64_extended_atomics",
        "cl_khr_il_program",
        "cl_khr_integer_dot_product",
        "cl_khr_extended_bit_ops",
        "cl_khr_fp16",
        "cl_ext_buffer_device_address",
        "cl_khr_kernel_clock",
        "cl_khr_subgroups",
        "cl_khr_fp64",
        "cl_nv_device_attribute_query",
    }
)
EXPECTED_FEATURES = frozenset(
    {
        "__opencl_c_atomic_order_acq_rel",
        "__opencl_c_atomic_order_seq_cst",
        "__opencl_c_atomic_scope_device",
        "__opencl_c_fp16",
        "__opencl_c_fp64",
        "__opencl_c_int64",
        "__opencl_c_integer_dot_product_input_4x8bit",
        "__opencl_c_integer_dot_product_input_4x8bit_packed",
        "__opencl_c_kernel_clock_scope_device",
        "__opencl_c_kernel_clock_scope_work_group",
        "__opencl_c_kernel_clock_scope_sub_group",
        "__opencl_c_subgroups",
        "__opencl_c_work_group_collective_functions",
    }
)
EXTENSION_VERSIONS = {
    "cl_khr_integer_dot_product": "2.0.0",
    "cl_ext_buffer_device_address": "1.0.2",
    "cl_khr_il_program": "2.1.0",
}
EXPECTED_EXTENSION_VERSIONS = frozenset(
    f"{name}@{EXTENSION_VERSIONS.get(name, '1.0.0')}" for name in EXPECTED_EXTENSIONS
)
EXPECTED_FEATURE_VERSIONS = frozenset(f"{name}@3.0.0" for name in EXPECTED_FEATURES)
EXPECTED_CONFORMANCE_VERSION = ""
CUDA_KERNEL_PARAMETER_BYTES = 4352
CUDA_IMPLICIT_PARAMETER_COUNT = 16
CUDA_IMPLICIT_PARAMETER_BYTES = 4
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
CUDA_HALF_FP_CONFIG = CL_FP_INF_NAN | CL_FP_ROUND_TO_NEAREST
EXPECTED_SCALARS = {
    "image_support": 0,
    "svm_capabilities": 1,
    "queue_properties": 1 << 1,
    "max_parameter_size": (
        CUDA_KERNEL_PARAMETER_BYTES
        - CUDA_IMPLICIT_PARAMETER_COUNT * CUDA_IMPLICIT_PARAMETER_BYTES
    ),
    "max_num_sub_groups": 32,
    "subgroup_forward_progress": 1,
    "atomic_memory_capabilities": (
        (1 << 0) | (1 << 1) | (1 << 2) | (1 << 4) | (1 << 5)
    ),
    "atomic_fence_capabilities": (
        (1 << 0) | (1 << 1) | (1 << 2) | (1 << 4) | (1 << 5)
    ),
    "non_uniform_support": 1,
    "work_group_collective_support": 1,
    "generic_address_support": 0,
    "pipe_support": 0,
    "single_fp_config": CUDA_SINGLE_FP_CONFIG,
    "double_fp_config": CUDA_DOUBLE_FP_CONFIG,
    "half_fp_config": CUDA_HALF_FP_CONFIG,
    "preferred_vector_width_half": 1,
    "native_vector_width_half": 1,
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
        "il_version": "SPIR-V_1.0",
        "latest_conformance": EXPECTED_CONFORMANCE_VERSION,
    }
    for field, expected in expected_strings.items():
        actual = getattr(device, field)
        if actual != expected:
            failures.append(f"{field}: expected {expected!r}, found {actual!r}")
    if (
        "OpenCL 3.0" not in device.device_version
        or "CUDA-sm_89-v9" not in device.device_version
    ):
        failures.append(
            "device_version does not identify OpenCL 3.0 CUDA-sm_89-v9: "
            f"{device.device_version}"
        )
    if device.il_versions != ("SPIR-V@1.0.0",):
        failures.append(
            f"il_versions: expected ['SPIR-V@1.0.0'], found {list(device.il_versions)}"
        )
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
    if frozenset(device.extension_versions) != EXPECTED_EXTENSION_VERSIONS:
        failures.append(
            "extension versions: expected "
            f"{sorted(EXPECTED_EXTENSION_VERSIONS)}, found {sorted(device.extension_versions)}"
        )
    if features != EXPECTED_FEATURES:
        failures.append(f"features: expected {sorted(EXPECTED_FEATURES)}, found {sorted(features)}")
    if frozenset(device.feature_versions) != EXPECTED_FEATURE_VERSIONS:
        failures.append(
            "feature versions: expected "
            f"{sorted(EXPECTED_FEATURE_VERSIONS)}, found {sorted(device.feature_versions)}"
        )
    if device.integer_dot_product_capabilities != 3:
        failures.append(
            "integer_dot_product_capabilities: expected packed and unpacked (3), "
            f"found {device.integer_dot_product_capabilities}"
        )
    if any(device.integer_dot_product_acceleration_8bit):
        failures.append("8-bit integer dot product acceleration was not established")
    if any(device.integer_dot_product_acceleration_4x8bit_packed):
        failures.append("packed integer dot product acceleration was not established")
    if device.kernel_clock_capabilities != 7:
        failures.append(
            "kernel_clock_capabilities: expected device, work-group, and "
            f"sub-group scopes (7), found {device.kernel_clock_capabilities}"
        )
    if device.half_fp_config != CUDA_HALF_FP_CONFIG or device.half_fp_query_error != 0:
        failures.append(
            "CL_DEVICE_HALF_FP_CONFIG must expose round-to-nearest and INF/NAN "
            f"support; value={device.half_fp_config}, error={device.half_fp_query_error}"
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
