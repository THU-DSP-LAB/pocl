#!/usr/bin/env python3

from __future__ import annotations

import unittest
from pathlib import Path

from cts_goal3_manifest import (
    CONVERSION_TEST_COUNT,
    MATH_NA_FUNCTIONS,
    MATH_VECTOR_WIDTHS,
    ManifestRow,
    audit_manifest,
    expected_conversion_ranges,
    math_pairs,
    reject_reduced_modes,
)

SCRIPT_DIR = Path(__file__).resolve().parent
MANIFEST = SCRIPT_DIR / "opencl_conformance_tests_goal3.csv"
MATH_FUNCTIONS = SCRIPT_DIR / "opencl_conformance_tests_goal3_math_functions.txt"


class Goal3ManifestTests(unittest.TestCase):
    def test_versioned_manifest_is_complete_and_unique(self) -> None:
        audit = audit_manifest(MANIFEST, MATH_FUNCTIONS)
        self.assertEqual(audit.manifest_rows, 307)
        self.assertEqual(audit.conversion_shards, 105)
        self.assertEqual(audit.conversion_tests, CONVERSION_TEST_COUNT)
        self.assertEqual(audit.registered_math_functions, 103)
        self.assertEqual(audit.applicable_math_functions, 101)
        self.assertEqual(audit.math_na_functions, tuple(sorted(MATH_NA_FUNCTIONS)))
        self.assertEqual(audit.math_shards, 202)
        self.assertEqual(audit.math_vector_widths, MATH_VECTOR_WIDTHS)

    def test_conversion_ranges_are_continuous_without_overlap(self) -> None:
        ranges = expected_conversion_ranges()
        flattened = tuple(
            number
            for start, count in ranges
            for number in range(start, start + count)
        )
        self.assertEqual(flattened, tuple(range(1, CONVERSION_TEST_COUNT + 1)))

    def test_math_rows_cannot_restrict_vector_width(self) -> None:
        row = ManifestRow(
            name="Math FP64 Full rootn",
            executable="math_brute_force/test_bruteforce",
            arguments=("-f", "-1", "rootn"),
        )
        with self.assertRaisesRegex(RuntimeError, "restricts more than precision"):
            math_pairs((row,))

    def test_wimpy_mode_is_rejected(self) -> None:
        row = ManifestRow(
            name="Conversions reduced",
            executable="conversions/test_conversions",
            arguments=("-w", "1", "10"),
        )
        with self.assertRaisesRegex(RuntimeError, "Reduced CTS mode"):
            reject_reduced_modes((row,))


if __name__ == "__main__":
    unittest.main()
