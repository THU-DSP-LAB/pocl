#!/usr/bin/env python3

from __future__ import annotations

import unittest
from pathlib import Path

from run_cts_quick import (
    marker_lines,
    selected_cache_dir,
    suite_csv_path,
    suite_was_skipped,
)


class SuiteSelectionTests(unittest.TestCase):
    def test_cache_defaults_inside_output_directory(self) -> None:
        output = Path("results").resolve()
        self.assertEqual(selected_cache_dir(output, None), output / "kernel-cache")

    def test_custom_cache_is_resolved(self) -> None:
        custom = Path("shared-cache")
        self.assertEqual(selected_cache_dir(Path("results"), custom), custom.resolve())

    def test_quick_manifest_comes_from_build_tree(self) -> None:
        path = suite_csv_path(
            Path("/scripts"), Path("/build"), "quick", custom_csv=None
        )
        self.assertEqual(
            path,
            Path(
                "/build/examples/conformance/src/conformance-build/"
                "test_conformance/opencl_conformance_tests_quick.csv"
            ),
        )

    def test_expanded_manifest_comes_from_script_tree(self) -> None:
        path = suite_csv_path(
            Path("/scripts"), Path("/build"), "expanded", custom_csv=None
        )
        self.assertEqual(path, Path("/scripts/opencl_conformance_tests_expanded.csv"))

    def test_custom_manifest_is_resolved(self) -> None:
        custom = Path("custom.csv")
        path = suite_csv_path(
            Path("/scripts"), Path("/build"), "quick", custom_csv=custom
        )
        self.assertEqual(path, custom.resolve())

    def test_expanded_manifest_has_unique_bounded_entries(self) -> None:
        manifest = Path(__file__).with_name("opencl_conformance_tests_expanded.csv")
        entries = tuple(
            line.strip()
            for line in manifest.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        )
        names = tuple(entry.split(",", 1)[0] for entry in entries)
        self.assertEqual(len(entries), 34)
        self.assertEqual(len(names), len(set(names)))
        self.assertEqual(sum(name.startswith("Vectors ") for name in names), 9)
        self.assertEqual(sum(name.startswith("Integer Ops ") for name in names), 5)
        self.assertEqual(sum(name.startswith("Math rootn ") for name in names), 7)
        self.assertIn(
            "Conversions Wimpy 1-523,conversions/test_conversions -w 1 523",
            entries,
        )
        self.assertIn(
            "Conversions Wimpy 524-1045,"
            "conversions/test_conversions -w 524 522",
            entries,
        )

    def test_integer_ops_partition_is_complete_and_unique(self) -> None:
        manifest = Path(__file__).with_name("opencl_conformance_tests_expanded.csv")
        rows = tuple(
            line.strip().split(",", 1)
            for line in manifest.read_text(encoding="utf-8").splitlines()
            if line.startswith("Integer Ops ")
        )
        arguments = tuple(row[1].split()[1:] for row in rows)
        flattened = tuple(name for group in arguments for name in group)
        self.assertEqual(tuple(len(group) for group in arguments), (16, 16, 16, 16, 35))
        self.assertEqual(len(flattened), 99)
        self.assertEqual(len(set(flattened)), len(flattened))
        self.assertEqual(arguments[0][0], "long_math")
        self.assertEqual(arguments[3][-1], "quick_uchar_compare")
        self.assertEqual(arguments[4][0], "vector_scalar")
        self.assertEqual(arguments[4][-1], "extended_bit_ops_reverse")


class OutputClassificationTests(unittest.TestCase):
    def test_informational_max_error_is_not_failure(self) -> None:
        output = "degrees: Max error 1.4 ulps\nPASSED 18 of 18 tests.\n"
        self.assertEqual(marker_lines(output), ())

    def test_cts_error_is_failure(self) -> None:
        output = "ERROR: clBuildProgram failed!\nfoo FAILED\n"
        self.assertEqual(len(marker_lines(output)), 2)

    def test_expected_build_error_is_not_failure(self) -> None:
        output = (
            "Build error detected at clBuildProgram."
            "Build failed as expected with #error in source:\n"
            "error: fictitious/file/name.c:124:2: some error\n"
            "Device GPU failed to build the program\n"
            "PASSED 72 of 72 tests.\n"
        )
        self.assertEqual(marker_lines(output), ())

    def test_internal_skip_does_not_skip_suite(self) -> None:
        output = "Skipping slow case.\nquick_3d passed\nPASSED 6 of 6 tests.\n"
        self.assertFalse(suite_was_skipped(output))

    def test_initialization_skip_skips_suite(self) -> None:
        output = "Test skipped while initialization\nSKIPPED 12 of 12 tests.\n"
        self.assertTrue(suite_was_skipped(output))

    def test_gl_unsupported_skips_suite(self) -> None:
        output = "Test not run because GL-CL interop is not supported.\n"
        self.assertTrue(suite_was_skipped(output))


if __name__ == "__main__":
    unittest.main()
