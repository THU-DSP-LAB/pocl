#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import os
import shlex
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

CONVERSION_EXECUTABLE = "conversions/test_conversions"
MATH_EXECUTABLE = "math_brute_force/test_bruteforce"
CONVERSION_TEST_COUNT = 1045
CONVERSION_SHARD_SIZE = 10
MATH_NA_FUNCTIONS = frozenset(("divide_cr", "sqrt_cr"))
MATH_PRECISION_OPTIONS = {"-d": "FP32", "-f": "FP64"}
MATH_VECTOR_WIDTHS = (1, 2, 3, 4, 8, 16)
CL_FP_CORRECTLY_ROUNDED_DIVIDE_SQRT = 1 << 7
LIST_TIMEOUT_SECONDS = 30


@dataclass(frozen=True)
class ManifestRow:
    name: str
    executable: str
    arguments: tuple[str, ...]


@dataclass(frozen=True)
class ManifestAudit:
    manifest_rows: int
    conversion_shards: int
    conversion_tests: int
    registered_math_functions: int
    applicable_math_functions: int
    math_na_functions: tuple[str, ...]
    math_shards: int
    math_vector_widths: tuple[int, ...]


def data_lines(path: Path) -> tuple[str, ...]:
    return tuple(
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    )


def load_math_functions(path: Path) -> tuple[str, ...]:
    functions = data_lines(path)
    if len(functions) != len(set(functions)):
        raise RuntimeError("Math function list contains duplicates")
    if not MATH_NA_FUNCTIONS.issubset(functions):
        raise RuntimeError("Math function list is missing capability N/A entries")
    return functions


def load_manifest(path: Path) -> tuple[ManifestRow, ...]:
    rows: list[ManifestRow] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for csv_row in csv.reader(handle):
            if not csv_row or csv_row[0].lstrip().startswith("#"):
                continue
            if len(csv_row) != 2:
                raise RuntimeError(f"Invalid Goal 3 CSV row: {csv_row}")
            tokens = shlex.split(csv_row[1].strip())
            if not tokens:
                raise RuntimeError(f"Empty Goal 3 command: {csv_row}")
            rows.append(
                ManifestRow(
                    name=csv_row[0].strip(),
                    executable=tokens[0],
                    arguments=tuple(tokens[1:]),
                )
            )
    if not rows:
        raise RuntimeError("Goal 3 manifest is empty")
    return tuple(rows)


def expected_conversion_ranges() -> tuple[tuple[int, int], ...]:
    return tuple(
        (start, min(CONVERSION_SHARD_SIZE, CONVERSION_TEST_COUNT + 1 - start))
        for start in range(1, CONVERSION_TEST_COUNT + 1, CONVERSION_SHARD_SIZE)
    )


def conversion_ranges(rows: tuple[ManifestRow, ...]) -> tuple[tuple[int, int], ...]:
    ranges: list[tuple[int, int]] = []
    for row in rows:
        if row.executable != CONVERSION_EXECUTABLE:
            continue
        if len(row.arguments) != 2:
            raise RuntimeError(f"Invalid Conversions shard: {row}")
        try:
            start, count = (int(value) for value in row.arguments)
        except ValueError as error:
            raise RuntimeError(f"Non-numeric Conversions shard: {row}") from error
        end = start + count - 1
        expected_name = f"Conversions Full {start:04d}-{end:04d}"
        if row.name != expected_name:
            raise RuntimeError(f"Unexpected Conversions shard name: {row.name}")
        ranges.append((start, count))
    return tuple(ranges)


def math_pairs(rows: tuple[ManifestRow, ...]) -> tuple[tuple[str, str], ...]:
    pairs: list[tuple[str, str]] = []
    for row in rows:
        if row.executable != MATH_EXECUTABLE:
            continue
        if len(row.arguments) != 2:
            raise RuntimeError(f"Math shard restricts more than precision: {row}")
        option, function = row.arguments
        precision = MATH_PRECISION_OPTIONS.get(option)
        if precision is None:
            raise RuntimeError(f"Invalid Math precision selector: {row}")
        if row.name != f"Math {precision} Full {function}":
            raise RuntimeError(f"Unexpected Math shard name: {row.name}")
        pairs.append((precision, function))
    return tuple(pairs)


def reject_reduced_modes(rows: tuple[ManifestRow, ...]) -> None:
    for row in rows:
        if "-w" in row.arguments or "--wimpy" in row.arguments:
            raise RuntimeError(f"Reduced CTS mode is forbidden: {row}")
        if row.executable not in (CONVERSION_EXECUTABLE, MATH_EXECUTABLE):
            raise RuntimeError(f"Unexpected Goal 3 executable: {row.executable}")


def audit_manifest(manifest_path: Path, function_path: Path) -> ManifestAudit:
    rows = load_manifest(manifest_path)
    functions = load_math_functions(function_path)
    applicable = tuple(name for name in functions if name not in MATH_NA_FUNCTIONS)
    expected_math = tuple(
        (precision, function)
        for precision in MATH_PRECISION_OPTIONS.values()
        for function in applicable
    )
    names = tuple(row.name for row in rows)
    commands = tuple((row.executable, row.arguments) for row in rows)
    if len(names) != len(set(names)) or len(commands) != len(set(commands)):
        raise RuntimeError("Goal 3 manifest contains duplicate names or commands")
    reject_reduced_modes(rows)
    ranges = conversion_ranges(rows)
    if ranges != expected_conversion_ranges():
        raise RuntimeError("Conversions shards are not an exact 1-1045 partition")
    pairs = math_pairs(rows)
    if pairs != expected_math:
        raise RuntimeError("Math shards do not exactly cover both precisions")
    return ManifestAudit(
        manifest_rows=len(rows),
        conversion_shards=len(ranges),
        conversion_tests=sum(count for _, count in ranges),
        registered_math_functions=len(functions),
        applicable_math_functions=len(applicable),
        math_na_functions=tuple(sorted(MATH_NA_FUNCTIONS)),
        math_shards=len(pairs),
        math_vector_widths=MATH_VECTOR_WIDTHS,
    )


def listed_tests(executable: Path, environment: dict[str, str]) -> tuple[str, ...]:
    completed = subprocess.run(
        (str(executable), "--list"),
        check=True,
        capture_output=True,
        text=True,
        timeout=LIST_TIMEOUT_SECONDS,
        env=environment,
    )
    return tuple(line.strip() for line in completed.stdout.splitlines() if line.strip())


def verify_cts_lists(
    cts_dir: Path,
    install_dir: Path,
    expected_math_functions: tuple[str, ...],
) -> None:
    environment = dict(os.environ)
    environment["LD_LIBRARY_PATH"] = str(install_dir / "lib")
    conversions = listed_tests(cts_dir / CONVERSION_EXECUTABLE, environment)
    math_functions = listed_tests(cts_dir / MATH_EXECUTABLE, environment)
    if len(conversions) != CONVERSION_TEST_COUNT:
        raise RuntimeError(f"CTS lists {len(conversions)} Conversions tests")
    if set(math_functions) != set(expected_math_functions):
        raise RuntimeError("Pinned Math list differs from the CTS executable")


def verify_device_contract(contract_path: Path) -> None:
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    fp_config = int(contract["single_fp_config"])
    if fp_config & CL_FP_CORRECTLY_ROUNDED_DIVIDE_SQRT:
        raise RuntimeError("divide_cr and sqrt_cr are applicable on this device")
    if contract.get("half_fp_config") != 0:
        raise RuntimeError("Goal 3 fixed baseline unexpectedly exposes fp16")


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Audit the Goal 3 CTS manifest")
    parser.add_argument(
        "--manifest", type=Path, default=script_dir / "opencl_conformance_tests_goal3.csv"
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
    functions = load_math_functions(args.math_functions)
    if args.cts_dir is not None:
        verify_cts_lists(args.cts_dir, args.install_dir, functions)
    if args.device_contract is not None:
        verify_device_contract(args.device_contract)
    print(json.dumps(asdict(audit), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
