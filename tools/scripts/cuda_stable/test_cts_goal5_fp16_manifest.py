#!/usr/bin/env python3

from __future__ import annotations

import unittest
from pathlib import Path

from cts_goal3_manifest import ManifestRow, load_manifest
from cts_goal5_fp16_manifest import (
    CONVERSION_EXECUTABLE,
    EXPECTED_CONVERSION_RANGES,
    EXPECTED_FP16_CONVERSION_TESTS,
    EXPECTED_MATH_FUNCTIONS,
    MATH_EXECUTABLE,
    MATH_VECTOR_WIDTHS,
    audit_manifest,
    expected_conversion_indices,
    math_rows,
    reject_reduced_or_unexpected_rows,
)

SCRIPT_DIR = Path(__file__).resolve().parent
MANIFEST = SCRIPT_DIR / "opencl_conformance_tests_goal5_fp16_full.csv"
REGRESSION_MANIFEST = (
    SCRIPT_DIR / "opencl_conformance_tests_goal5_fp16_regressions.csv"
)
MATH_FUNCTIONS = SCRIPT_DIR / "opencl_conformance_tests_goal3_math_functions.txt"
EXPECTED_REGRESSION_NAMES = (
    "FP16 Conversions Full 0681-0690",
    "FP16 Conversions Full 0711-0715",
    "FP16 Math Full fmod",
    "FP16 Math Full isfinite",
    "FP16 Math Full isinf",
    "FP16 Math Full isnan",
    "FP16 Math Full isnormal",
    "FP16 Math Full modf",
)


class Goal5FP16ManifestTests(unittest.TestCase):
    def test_versioned_manifest_is_complete_and_unique(self) -> None:
        audit = audit_manifest(MANIFEST, MATH_FUNCTIONS)
        self.assertEqual(audit.manifest_rows, 103)
        self.assertEqual(audit.conversion_shards, len(EXPECTED_CONVERSION_RANGES))
        self.assertEqual(audit.conversion_tests, EXPECTED_FP16_CONVERSION_TESTS)
        self.assertEqual(audit.math_functions, EXPECTED_MATH_FUNCTIONS)
        self.assertEqual(audit.math_shards, EXPECTED_MATH_FUNCTIONS)
        self.assertEqual(audit.math_vector_widths, MATH_VECTOR_WIDTHS)

    def test_conversion_ranges_cover_only_fp16_tests(self) -> None:
        indices = expected_conversion_indices()
        self.assertEqual(len(indices), EXPECTED_FP16_CONVERSION_TESTS)
        self.assertEqual(len(indices), len(set(indices)))

    def test_math_rows_cannot_restrict_vector_width(self) -> None:
        row = ManifestRow(
            name="FP16 Math Full acos",
            executable=MATH_EXECUTABLE,
            arguments=("-f", "-d", "-r", "-3", "acos"),
        )
        with self.assertRaisesRegex(RuntimeError, "restricts vector widths"):
            math_rows((row,))

    def test_wimpy_mode_is_rejected(self) -> None:
        row = ManifestRow(
            name="FP16 Conversions reduced",
            executable=CONVERSION_EXECUTABLE,
            arguments=("-w", "61", "10"),
        )
        with self.assertRaisesRegex(RuntimeError, "Reduced CTS mode"):
            reject_reduced_or_unexpected_rows((row,))

    def test_regressions_are_exact_full_manifest_rows(self) -> None:
        full_rows = load_manifest(MANIFEST)
        regression_rows = load_manifest(REGRESSION_MANIFEST)
        self.assertEqual(
            tuple(row.name for row in regression_rows),
            EXPECTED_REGRESSION_NAMES,
        )
        self.assertEqual(len(regression_rows), len(set(regression_rows)))
        self.assertTrue(set(regression_rows).issubset(full_rows))
        reject_reduced_or_unexpected_rows(regression_rows)


if __name__ == "__main__":
    unittest.main()
