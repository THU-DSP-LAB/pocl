#!/usr/bin/env python3

from __future__ import annotations

import unittest

from run_cts_quick import marker_lines, suite_was_skipped


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
