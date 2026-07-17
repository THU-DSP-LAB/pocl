#!/usr/bin/env python3

from __future__ import annotations

import re
from dataclasses import dataclass

REGISTERED_TEST_START_PATTERN = re.compile(
    r"^(?P<name>[A-Za-z0-9_./-]+)\.\.\.$"
)
AGGREGATE_PASS_PATTERN = re.compile(r"^PASSED (\d+) of (\d+) tests\.$")
NOT_SUPPORTED_SUFFIX = " test not supported"
PASSED_SUFFIX = " passed"
UNSUPPORTED_REASON_MARKERS = ("not support", "skipping")


@dataclass(frozen=True)
class OutputAssessment:
    selected_test_count: int
    applicable_pass_count: int
    not_supported_count: int
    not_supported_tests: tuple[str, ...]
    not_supported_reasons: tuple[str, ...]
    all_selected_not_supported: bool


@dataclass(frozen=True)
class RegisteredResults:
    started: tuple[str, ...]
    passed: tuple[str, ...]
    not_supported: tuple[str, ...]
    reasons: tuple[str, ...]


def append_unique(items: list[str], value: str) -> None:
    if value not in items:
        items.append(value)


def is_unsupported_reason(line: str) -> bool:
    lowered = line.lower()
    return any(marker in lowered for marker in UNSUPPORTED_REASON_MARKERS)


def registered_results(output: str) -> RegisteredResults:
    started: list[str] = []
    passed: list[str] = []
    not_supported: list[str] = []
    reasons: list[str] = []
    active_test: str | None = None
    active_reason: str | None = None

    for raw_line in output.splitlines():
        line = raw_line.strip()
        start_match = REGISTERED_TEST_START_PATTERN.fullmatch(line)
        if start_match is not None:
            active_test = start_match.group("name")
            active_reason = None
            append_unique(started, active_test)
            continue
        if active_test is None:
            continue
        if line == active_test + PASSED_SUFFIX:
            append_unique(passed, active_test)
            active_test = None
            active_reason = None
            continue
        if line != active_test + NOT_SUPPORTED_SUFFIX:
            if is_unsupported_reason(line):
                active_reason = line
            continue
        append_unique(not_supported, active_test)
        if active_reason is not None:
            append_unique(reasons, active_reason)
        active_test = None
        active_reason = None

    return RegisteredResults(
        started=tuple(started),
        passed=tuple(passed),
        not_supported=tuple(not_supported),
        reasons=tuple(reasons),
    )


def aggregate_pass_count(output: str) -> int | None:
    counts = tuple(
        int(match.group(1))
        for line in output.splitlines()
        if (match := AGGREGATE_PASS_PATTERN.fullmatch(line.strip())) is not None
    )
    return counts[-1] if counts else None


def assess_output(output: str) -> OutputAssessment:
    results = registered_results(output)
    aggregate_count = aggregate_pass_count(output)
    applicable_pass_count = (
        aggregate_count if aggregate_count is not None else len(results.passed)
    )
    not_supported_count = len(results.not_supported)
    all_results_accounted_for = len(results.started) == not_supported_count
    same_documented_reason = len(results.reasons) == 1
    all_selected_not_supported = (
        not_supported_count > 0
        and applicable_pass_count == 0
        and all_results_accounted_for
        and same_documented_reason
    )
    return OutputAssessment(
        selected_test_count=applicable_pass_count + not_supported_count,
        applicable_pass_count=applicable_pass_count,
        not_supported_count=not_supported_count,
        not_supported_tests=results.not_supported,
        not_supported_reasons=results.reasons,
        all_selected_not_supported=all_selected_not_supported,
    )
