#!/usr/bin/env python3

from __future__ import annotations

import unittest
from pathlib import Path

from run_cts_quick import (
    assess_output,
    classify_result,
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

    def test_goal3_manifest_comes_from_script_tree(self) -> None:
        path = suite_csv_path(
            Path("/scripts"), Path("/build"), "goal3", custom_csv=None
        )
        self.assertEqual(path, Path("/scripts/opencl_conformance_tests_goal3.csv"))

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
    ALL_UNSUPPORTED_OUTPUT = """\
cxx_for_opencl_ext...
Device does not support 'cl_ext_cxx_for_opencl'. Skipping the test.
cxx_for_opencl_ext test not supported
cxx_for_opencl_ver...
Device does not support 'cl_ext_cxx_for_opencl'. Skipping the test.
cxx_for_opencl_ver test not supported
PASSED sub-test.
PASSED test.
"""

    MIXED_INTEGER_OUTPUT = """\
popcount...
popcount passed
integer_dot_product...
cl_khr_integer_dot_product is not supported
integer_dot_product test not supported
extended_bit_ops_extract...
cl_khr_extended_bit_ops is not supported
extended_bit_ops_extract test not supported
extended_bit_ops_insert...
cl_khr_extended_bit_ops is not supported
extended_bit_ops_insert test not supported
extended_bit_ops_reverse...
cl_khr_extended_bit_ops is not supported
extended_bit_ops_reverse test not supported
PASSED sub-test.
PASSED 31 of 31 tests.
"""

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

    def test_all_registered_tests_unsupported_is_auditable_skip(self) -> None:
        assessment = assess_output(self.ALL_UNSUPPORTED_OUTPUT)
        self.assertTrue(assessment.all_selected_not_supported)
        self.assertEqual(assessment.applicable_pass_count, 0)
        self.assertEqual(
            assessment.not_supported_tests,
            ("cxx_for_opencl_ext", "cxx_for_opencl_ver"),
        )
        self.assertEqual(
            assessment.not_supported_reasons,
            ("Device does not support 'cl_ext_cxx_for_opencl'. Skipping the test.",),
        )

    def test_mixed_supported_and_na_tests_remains_applicable(self) -> None:
        assessment = assess_output(self.MIXED_INTEGER_OUTPUT)
        self.assertFalse(assessment.all_selected_not_supported)
        self.assertEqual(assessment.applicable_pass_count, 31)
        self.assertEqual(assessment.not_supported_count, 4)
        self.assertEqual(len(assessment.not_supported_reasons), 2)

    def test_unpaired_unsupported_text_does_not_skip_suite(self) -> None:
        assessment = assess_output("feature test not supported\nPASSED test.\n")
        self.assertFalse(assessment.all_selected_not_supported)
        self.assertEqual(assessment.not_supported_count, 0)

    def test_failures_take_priority_over_all_na(self) -> None:
        common = {
            "crash_signal": None,
            "return_code": 0,
            "failure_markers": (),
            "suite_skipped": False,
            "all_selected_not_supported": True,
        }
        self.assertEqual(classify_result(timed_out=True, **common), "timeout")
        self.assertEqual(
            classify_result(
                timed_out=False,
                **{**common, "crash_signal": 11, "return_code": -11},
            ),
            "crash",
        )
        self.assertEqual(
            classify_result(
                timed_out=False,
                **{**common, "return_code": 1},
            ),
            "fail",
        )
        self.assertEqual(
            classify_result(
                timed_out=False,
                **{**common, "failure_markers": ("ERROR: build failed",)},
            ),
            "fail",
        )


if __name__ == "__main__":
    unittest.main()
