#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

from cts_goal3_manifest import (
    CONVERSION_EXECUTABLE,
    CONVERSION_TEST_COUNT,
    MATH_EXECUTABLE,
    MATH_VECTOR_WIDTHS,
    ManifestRow,
    data_lines,
    listed_tests,
    load_manifest,
)

CONVERSION_SHARD_SIZE = 10
EXPECTED_FP16_CONVERSION_TESTS = 145
EXPECTED_HALF_FP_CONFIG = 6
EXPECTED_HALF_VECTOR_WIDTH = 1
EXPECTED_MATH_FUNCTIONS = 87
FP16_EXTENSION = "cl_khr_fp16"
FP16_FEATURE = "__opencl_c_fp16"
MATH_OPTIONS = ("-f", "-d", "-r")
MATH_EXCLUDED_FUNCTIONS = frozenset(
    (
        "divide_cr",
        "sqrt_cr",
        "half_cos",
        "half_divide",
        "half_exp",
        "half_exp2",
        "half_exp10",
        "half_log",
        "half_log2",
        "half_log10",
        "half_powr",
        "half_recip",
        "half_rsqrt",
        "half_sin",
        "half_sqrt",
        "half_tan",
    )
)
EXPECTED_CONVERSION_RANGES = (
    (61, 10),
    (171, 10),
    (281, 10),
    (391, 10),
    (501, 10),
    (611, 10),
    (661, 10),
    (671, 10),
    (681, 10),
    (691, 10),
    (701, 10),
    (711, 5),
    (746, 5),
    (801, 5),
    (886, 10),
    (996, 10),
)


@dataclass(frozen=True)
class ManifestAudit:
    manifest_rows: int
    conversion_shards: int
    conversion_tests: int
    math_functions: int
    math_shards: int
    math_vector_widths: tuple[int, ...]


def expected_conversion_indices() -> tuple[int, ...]:
    return tuple(
        index
        for start, count in EXPECTED_CONVERSION_RANGES
        for index in range(start, start + count)
    )


def fp16_conversion_indices(names: tuple[str, ...]) -> tuple[int, ...]:
    return tuple(
        index
        for index, name in enumerate(names, start=1)
        if "half" in name
    )


def conversion_ranges(rows: tuple[ManifestRow, ...]) -> tuple[tuple[int, int], ...]:
    ranges: list[tuple[int, int]] = []
    for row in rows:
        if row.executable != CONVERSION_EXECUTABLE:
            continue
        if len(row.arguments) != 2:
            raise RuntimeError(f"Invalid FP16 Conversions shard: {row}")
        try:
            start, count = (int(value) for value in row.arguments)
        except ValueError as error:
            raise RuntimeError(f"Non-numeric FP16 Conversions shard: {row}") from error
        end = start + count - 1
        expected_name = f"FP16 Conversions Full {start:04d}-{end:04d}"
        if row.name != expected_name or count > CONVERSION_SHARD_SIZE:
            raise RuntimeError(f"Unexpected FP16 Conversions shard: {row}")
        ranges.append((start, count))
    return tuple(ranges)


def applicable_math_functions(path: Path) -> tuple[str, ...]:
    registered = data_lines(path)
    functions = tuple(
        name for name in registered if name not in MATH_EXCLUDED_FUNCTIONS
    )
    if len(functions) != EXPECTED_MATH_FUNCTIONS:
        raise RuntimeError(f"Expected {EXPECTED_MATH_FUNCTIONS} FP16 Math functions")
    return functions


def math_rows(rows: tuple[ManifestRow, ...]) -> tuple[str, ...]:
    functions: list[str] = []
    for row in rows:
        if row.executable != MATH_EXECUTABLE:
            continue
        if len(row.arguments) != len(MATH_OPTIONS) + 1:
            raise RuntimeError(f"FP16 Math shard restricts vector widths: {row}")
        options = row.arguments[:-1]
        function = row.arguments[-1]
        if options != MATH_OPTIONS or row.name != f"FP16 Math Full {function}":
            raise RuntimeError(f"Unexpected FP16 Math shard: {row}")
        functions.append(function)
    return tuple(functions)


def reject_reduced_or_unexpected_rows(rows: tuple[ManifestRow, ...]) -> None:
    vector_options = {f"-{width}" for width in MATH_VECTOR_WIDTHS}
    for row in rows:
        if "-w" in row.arguments or "--wimpy" in row.arguments:
            raise RuntimeError(f"Reduced CTS mode is forbidden: {row}")
        if vector_options.intersection(row.arguments):
            raise RuntimeError(f"Math vector-width restriction is forbidden: {row}")
        if row.executable not in (CONVERSION_EXECUTABLE, MATH_EXECUTABLE):
            raise RuntimeError(f"Unexpected FP16 executable: {row.executable}")


def audit_manifest(manifest_path: Path, function_path: Path) -> ManifestAudit:
    rows = load_manifest(manifest_path)
    functions = applicable_math_functions(function_path)
    names = tuple(row.name for row in rows)
    commands = tuple((row.executable, row.arguments) for row in rows)
    if len(names) != len(set(names)) or len(commands) != len(set(commands)):
        raise RuntimeError("FP16 manifest contains duplicate names or commands")
    reject_reduced_or_unexpected_rows(rows)
    ranges = conversion_ranges(rows)
    if ranges != EXPECTED_CONVERSION_RANGES:
        raise RuntimeError("FP16 Conversions shards do not match the pinned CTS list")
    if math_rows(rows) != functions:
        raise RuntimeError("FP16 Math shards do not match the pinned CTS list")
    return ManifestAudit(
        manifest_rows=len(rows),
        conversion_shards=len(ranges),
        conversion_tests=sum(count for _, count in ranges),
        math_functions=len(functions),
        math_shards=len(functions),
        math_vector_widths=MATH_VECTOR_WIDTHS,
    )


def verify_cts_lists(
    cts_dir: Path, install_dir: Path, expected_math: tuple[str, ...]
) -> None:
    environment = dict(os.environ)
    environment["LD_LIBRARY_PATH"] = str(install_dir / "lib")
    conversions = listed_tests(cts_dir / CONVERSION_EXECUTABLE, environment)
    registered_math = listed_tests(cts_dir / MATH_EXECUTABLE, environment)
    if len(conversions) != CONVERSION_TEST_COUNT:
        raise RuntimeError(f"CTS lists {len(conversions)} Conversions tests")
    if fp16_conversion_indices(conversions) != expected_conversion_indices():
        raise RuntimeError("Pinned FP16 Conversions indices differ from the CTS list")
    live_math = tuple(
        name for name in registered_math if name not in MATH_EXCLUDED_FUNCTIONS
    )
    if live_math != expected_math:
        raise RuntimeError("Pinned FP16 Math functions differ from the CTS list")


def verify_device_contract(contract_path: Path) -> None:
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    if FP16_EXTENSION not in contract["extensions"]:
        raise RuntimeError("Device contract does not expose cl_khr_fp16")
    if FP16_FEATURE not in contract["features"]:
        raise RuntimeError("Device contract does not expose __opencl_c_fp16")
    if contract["half_fp_config"] != EXPECTED_HALF_FP_CONFIG:
        raise RuntimeError("Device contract has an unexpected half FP config")
    widths = (
        contract["preferred_vector_width_half"],
        contract["native_vector_width_half"],
    )
    if widths != (EXPECTED_HALF_VECTOR_WIDTH, EXPECTED_HALF_VECTOR_WIDTH):
        raise RuntimeError("Device contract has unexpected half vector widths")


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Audit the Goal 5 FP16 CTS manifest")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=script_dir / "opencl_conformance_tests_goal5_fp16_full.csv",
    )
    parser.add_argument(
        "--math-functions",
        type=Path,
        default=script_dir / "opencl_conformance_tests_goal3_math_functions.txt",
    )
    parser.add_argument("--cts-dir", type=Path)
    parser.add_argument("--install-dir", type=Path)
    parser.add_argument("--device-contract", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    audit = audit_manifest(args.manifest, args.math_functions)
    if (args.cts_dir is None) != (args.install_dir is None):
        raise RuntimeError("--cts-dir and --install-dir must be supplied together")
    functions = applicable_math_functions(args.math_functions)
    if args.cts_dir is not None:
        verify_cts_lists(args.cts_dir, args.install_dir, functions)
    if args.device_contract is not None:
        verify_device_contract(args.device_contract)
    print(json.dumps(asdict(audit), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
