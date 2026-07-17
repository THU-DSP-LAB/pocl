#!/usr/bin/env python3

from __future__ import annotations

import ctypes
from dataclasses import dataclass
from pathlib import Path

CL_SUCCESS = 0
CL_DEVICE_TYPE_GPU = 1 << 2
CL_NAME_VERSION_MAX_NAME_SIZE = 64

CL_DEVICE_IMAGE_SUPPORT = 0x1016
CL_DEVICE_MAX_PARAMETER_SIZE = 0x1017
CL_DEVICE_SINGLE_FP_CONFIG = 0x101B
CL_DEVICE_QUEUE_ON_HOST_PROPERTIES = 0x102A
CL_DEVICE_NAME = 0x102B
CL_DRIVER_VERSION = 0x102D
CL_DEVICE_VERSION = 0x102F
CL_DEVICE_EXTENSIONS = 0x1030
CL_DEVICE_HALF_FP_CONFIG = 0x1033
CL_DEVICE_DOUBLE_FP_CONFIG = 0x1032
CL_DEVICE_OPENCL_C_VERSION = 0x103D
CL_DEVICE_SVM_CAPABILITIES = 0x1053
CL_DEVICE_IL_VERSION = 0x105B
CL_DEVICE_MAX_NUM_SUB_GROUPS = 0x105C
CL_DEVICE_SUB_GROUP_INDEPENDENT_FORWARD_PROGRESS = 0x105D
CL_DEVICE_EXTENSIONS_WITH_VERSION = 0x1060
CL_DEVICE_ATOMIC_MEMORY_CAPABILITIES = 0x1063
CL_DEVICE_ATOMIC_FENCE_CAPABILITIES = 0x1064
CL_DEVICE_NON_UNIFORM_WORK_GROUP_SUPPORT = 0x1065
CL_DEVICE_GENERIC_ADDRESS_SPACE_SUPPORT = 0x1069
CL_DEVICE_OPENCL_C_FEATURES = 0x106F
CL_DEVICE_PIPE_SUPPORT = 0x1071
CL_DEVICE_LATEST_CONFORMANCE_VERSION_PASSED = 0x1072
CL_DEVICE_INTEGER_DOT_PRODUCT_CAPABILITIES_KHR = 0x1073
CL_DEVICE_INTEGER_DOT_PRODUCT_ACCELERATION_PROPERTIES_8BIT_KHR = 0x1074
CL_DEVICE_INTEGER_DOT_PRODUCT_ACCELERATION_PROPERTIES_4X8BIT_PACKED_KHR = 0x1075
CL_DEVICE_KERNEL_CLOCK_CAPABILITIES_KHR = 0x1076


class NameVersion(ctypes.Structure):
    _fields_ = [
        ("version", ctypes.c_uint32),
        ("name", ctypes.c_char * CL_NAME_VERSION_MAX_NAME_SIZE),
    ]


class DotProductAccelerationProperties(ctypes.Structure):
    _fields_ = [
        ("signed_accelerated", ctypes.c_uint),
        ("unsigned_accelerated", ctypes.c_uint),
        ("mixed_signedness_accelerated", ctypes.c_uint),
        ("accumulating_saturating_signed_accelerated", ctypes.c_uint),
        ("accumulating_saturating_unsigned_accelerated", ctypes.c_uint),
        ("accumulating_saturating_mixed_signedness_accelerated", ctypes.c_uint),
    ]


def decode_version(version: int) -> str:
    major = version >> 22
    minor = (version >> 12) & 0x3FF
    patch = version & 0xFFF
    return f"{major}.{minor}.{patch}"


@dataclass(frozen=True)
class DeviceSnapshot:
    library: str
    name: str
    driver_version: str
    device_version: str
    opencl_c_version: str
    extensions: tuple[str, ...]
    versioned_extensions: tuple[str, ...]
    extension_versions: tuple[str, ...]
    features: tuple[str, ...]
    feature_versions: tuple[str, ...]
    il_version: str
    latest_conformance: str
    image_support: int
    single_fp_config: int
    double_fp_config: int
    half_fp_config: int | None
    half_fp_query_error: int
    svm_capabilities: int
    queue_properties: int
    max_parameter_size: int
    max_num_sub_groups: int
    subgroup_forward_progress: int
    atomic_memory_capabilities: int
    atomic_fence_capabilities: int
    non_uniform_support: int
    generic_address_support: int
    pipe_support: int
    integer_dot_product_capabilities: int
    integer_dot_product_acceleration_8bit: tuple[int, ...]
    integer_dot_product_acceleration_4x8bit_packed: tuple[int, ...]
    kernel_clock_capabilities: int


class OpenCLProbe:
    def __init__(self, library: Path) -> None:
        self.library_path = library.resolve(strict=True)
        self.api = ctypes.CDLL(str(self.library_path))
        self._configure_api()
        self.device = self._get_single_gpu()

    def _configure_api(self) -> None:
        self.api.clGetPlatformIDs.argtypes = [
            ctypes.c_uint,
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(ctypes.c_uint),
        ]
        self.api.clGetPlatformIDs.restype = ctypes.c_int
        self.api.clGetDeviceIDs.argtypes = [
            ctypes.c_void_p,
            ctypes.c_ulong,
            ctypes.c_uint,
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(ctypes.c_uint),
        ]
        self.api.clGetDeviceIDs.restype = ctypes.c_int
        self.api.clGetDeviceInfo.argtypes = [
            ctypes.c_void_p,
            ctypes.c_uint,
            ctypes.c_size_t,
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_size_t),
        ]
        self.api.clGetDeviceInfo.restype = ctypes.c_int

    @staticmethod
    def _check(error: int, operation: str) -> None:
        if error != CL_SUCCESS:
            raise RuntimeError(f"{operation} failed with OpenCL error {error}")

    def _get_single_gpu(self) -> ctypes.c_void_p:
        platform_count = ctypes.c_uint()
        self._check(
            self.api.clGetPlatformIDs(0, None, ctypes.byref(platform_count)),
            "clGetPlatformIDs(count)",
        )
        platforms = (ctypes.c_void_p * platform_count.value)()
        self._check(
            self.api.clGetPlatformIDs(platform_count, platforms, None),
            "clGetPlatformIDs(list)",
        )
        devices: list[ctypes.c_void_p] = []
        for platform in platforms:
            device_count = ctypes.c_uint()
            error = self.api.clGetDeviceIDs(
                platform, CL_DEVICE_TYPE_GPU, 0, None, ctypes.byref(device_count)
            )
            if error != CL_SUCCESS:
                continue
            platform_devices = (ctypes.c_void_p * device_count.value)()
            self._check(
                self.api.clGetDeviceIDs(
                    platform,
                    CL_DEVICE_TYPE_GPU,
                    device_count,
                    platform_devices,
                    None,
                ),
                "clGetDeviceIDs(list)",
            )
            devices.extend(platform_devices)
        if len(devices) != 1:
            raise RuntimeError(f"Expected exactly one GPU device, found {len(devices)}")
        return devices[0]

    def string(self, parameter: int) -> str:
        size = ctypes.c_size_t()
        self._check(
            self.api.clGetDeviceInfo(self.device, parameter, 0, None, ctypes.byref(size)),
            f"clGetDeviceInfo({parameter:#x}, size)",
        )
        buffer = ctypes.create_string_buffer(size.value)
        self._check(
            self.api.clGetDeviceInfo(self.device, parameter, size, buffer, None),
            f"clGetDeviceInfo({parameter:#x}, value)",
        )
        return buffer.value.decode("utf-8")

    def scalar(self, parameter: int, scalar_type: type[ctypes._SimpleCData]) -> int:
        value, error = self.scalar_result(parameter, scalar_type)
        self._check(error, f"clGetDeviceInfo({parameter:#x})")
        if value is None:
            raise RuntimeError(f"clGetDeviceInfo({parameter:#x}) returned no value")
        return value

    def scalar_result(
        self, parameter: int, scalar_type: type[ctypes._SimpleCData]
    ) -> tuple[int | None, int]:
        value = scalar_type()
        error = self.api.clGetDeviceInfo(
            self.device,
            parameter,
            ctypes.sizeof(value),
            ctypes.byref(value),
            None,
        )
        return (int(value.value), error) if error == CL_SUCCESS else (None, error)

    def names(self, parameter: int) -> tuple[str, ...]:
        return tuple(
            sorted(
                item.name.decode("utf-8") for item in self.name_versions(parameter)
            )
        )

    def named_versions(self, parameter: int) -> tuple[str, ...]:
        return tuple(
            sorted(
                f"{item.name.decode('utf-8')}@{decode_version(item.version)}"
                for item in self.name_versions(parameter)
            )
        )

    def name_versions(self, parameter: int) -> tuple[NameVersion, ...]:
        size = ctypes.c_size_t()
        self._check(
            self.api.clGetDeviceInfo(self.device, parameter, 0, None, ctypes.byref(size)),
            f"clGetDeviceInfo({parameter:#x}, size)",
        )
        count = size.value // ctypes.sizeof(NameVersion)
        if count == 0:
            return ()
        values = (NameVersion * count)()
        self._check(
            self.api.clGetDeviceInfo(self.device, parameter, size, values, None),
            f"clGetDeviceInfo({parameter:#x}, value)",
        )
        return tuple(values)

    def structure_values(
        self, parameter: int, structure_type: type[ctypes.Structure]
    ) -> tuple[int, ...]:
        value = structure_type()
        self._check(
            self.api.clGetDeviceInfo(
                self.device,
                parameter,
                ctypes.sizeof(value),
                ctypes.byref(value),
                None,
            ),
            f"clGetDeviceInfo({parameter:#x})",
        )
        return tuple(int(getattr(value, field)) for field, _ in value._fields_)


def snapshot(library: Path) -> DeviceSnapshot:
    probe = OpenCLProbe(library)
    ulong = ctypes.c_ulong
    size_t = ctypes.c_size_t
    uint = ctypes.c_uint
    half_fp_config, half_fp_error = probe.scalar_result(CL_DEVICE_HALF_FP_CONFIG, ulong)
    return DeviceSnapshot(
        library=str(probe.library_path),
        name=probe.string(CL_DEVICE_NAME),
        driver_version=probe.string(CL_DRIVER_VERSION),
        device_version=probe.string(CL_DEVICE_VERSION),
        opencl_c_version=probe.string(CL_DEVICE_OPENCL_C_VERSION),
        extensions=tuple(sorted(probe.string(CL_DEVICE_EXTENSIONS).split())),
        versioned_extensions=probe.names(CL_DEVICE_EXTENSIONS_WITH_VERSION),
        extension_versions=probe.named_versions(CL_DEVICE_EXTENSIONS_WITH_VERSION),
        features=probe.names(CL_DEVICE_OPENCL_C_FEATURES),
        feature_versions=probe.named_versions(CL_DEVICE_OPENCL_C_FEATURES),
        il_version=probe.string(CL_DEVICE_IL_VERSION),
        latest_conformance=probe.string(CL_DEVICE_LATEST_CONFORMANCE_VERSION_PASSED),
        image_support=probe.scalar(CL_DEVICE_IMAGE_SUPPORT, uint),
        single_fp_config=probe.scalar(CL_DEVICE_SINGLE_FP_CONFIG, ulong),
        double_fp_config=probe.scalar(CL_DEVICE_DOUBLE_FP_CONFIG, ulong),
        half_fp_config=half_fp_config,
        half_fp_query_error=half_fp_error,
        svm_capabilities=probe.scalar(CL_DEVICE_SVM_CAPABILITIES, ulong),
        queue_properties=probe.scalar(CL_DEVICE_QUEUE_ON_HOST_PROPERTIES, ulong),
        max_parameter_size=probe.scalar(CL_DEVICE_MAX_PARAMETER_SIZE, size_t),
        max_num_sub_groups=probe.scalar(CL_DEVICE_MAX_NUM_SUB_GROUPS, uint),
        subgroup_forward_progress=probe.scalar(
            CL_DEVICE_SUB_GROUP_INDEPENDENT_FORWARD_PROGRESS, uint
        ),
        atomic_memory_capabilities=probe.scalar(
            CL_DEVICE_ATOMIC_MEMORY_CAPABILITIES, ulong
        ),
        atomic_fence_capabilities=probe.scalar(
            CL_DEVICE_ATOMIC_FENCE_CAPABILITIES, ulong
        ),
        non_uniform_support=probe.scalar(CL_DEVICE_NON_UNIFORM_WORK_GROUP_SUPPORT, uint),
        generic_address_support=probe.scalar(
            CL_DEVICE_GENERIC_ADDRESS_SPACE_SUPPORT, uint
        ),
        pipe_support=probe.scalar(CL_DEVICE_PIPE_SUPPORT, uint),
        integer_dot_product_capabilities=probe.scalar(
            CL_DEVICE_INTEGER_DOT_PRODUCT_CAPABILITIES_KHR, ulong
        ),
        integer_dot_product_acceleration_8bit=probe.structure_values(
            CL_DEVICE_INTEGER_DOT_PRODUCT_ACCELERATION_PROPERTIES_8BIT_KHR,
            DotProductAccelerationProperties,
        ),
        integer_dot_product_acceleration_4x8bit_packed=probe.structure_values(
            CL_DEVICE_INTEGER_DOT_PRODUCT_ACCELERATION_PROPERTIES_4X8BIT_PACKED_KHR,
            DotProductAccelerationProperties,
        ),
        kernel_clock_capabilities=probe.scalar(
            CL_DEVICE_KERNEL_CLOCK_CAPABILITIES_KHR, ulong
        ),
    )
