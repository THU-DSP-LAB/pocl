#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shlex
import signal
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from cts_output import assess_output

FAILURE_PATTERN = re.compile(
    r"(^\s*ERROR:|\bFAILED\b|\bAssertion\b|\bSIGFPE\b|\bSIGSEGV\b|"
    r"Segmentation fault|core dumped|CUDA_ERROR_|Cannot select)",
    re.MULTILINE,
)
SUITE_SKIP_PATTERNS = (
    re.compile(r"^\s*SKIPPED(?:\s+\d+\s+of\s+\d+\s+tests|:)", re.MULTILINE),
    re.compile(r"^\s*Test skipped while initialization\s*$", re.MULTILINE),
    re.compile(r"^\s*Test not run because\b", re.MULTILINE),
    re.compile(r"^\s*cl_khr_spir is not supported.*Skipping test\.$", re.MULTILINE),
)
TERMINATION_GRACE_SECONDS = 5
GPU_IDLE_WAIT_SECONDS = 10
GPU_IDLE_POLL_SECONDS = 0.2
DEFAULT_CTS_TIMEOUT_SECONDS = 900
DEFAULT_SUITE = "quick"
KNOWN_SUITES = (DEFAULT_SUITE, "expanded", "goal3")


@dataclass(frozen=True)
class TestCase:
    name: str
    executable: Path
    arguments: tuple[str, ...]


@dataclass(frozen=True)
class TestResult:
    name: str
    command: tuple[str, ...]
    status: str
    return_code: int | None
    signal: int | None
    duration_seconds: float
    timed_out: bool
    failure_markers: tuple[str, ...]
    selected_test_count: int
    applicable_pass_count: int
    not_supported_count: int
    not_supported_tests: tuple[str, ...]
    not_supported_reasons: tuple[str, ...]
    stdout_log: str
    stderr_log: str


@dataclass(frozen=True)
class RunnerConfig:
    build_dir: Path
    install_dir: Path
    output_dir: Path
    timeout_seconds: int | None
    name_filter: str | None
    suite_name: str
    csv_path: Path
    cache_dir: Path

    @property
    def cts_dir(self) -> Path:
        return self.build_dir / "examples/conformance/src/conformance-build/test_conformance"

    @property
    def library(self) -> Path:
        return self.install_dir / "lib/libOpenCL.so"


def suite_csv_path(
    script_dir: Path,
    build_dir: Path,
    suite_name: str,
    *,
    custom_csv: Path | None,
) -> Path:
    if custom_csv is not None:
        return custom_csv.resolve()
    if suite_name == DEFAULT_SUITE:
        return (
            build_dir
            / "examples/conformance/src/conformance-build/test_conformance"
            / "opencl_conformance_tests_quick.csv"
        )
    return script_dir / f"opencl_conformance_tests_{suite_name}.csv"


def selected_cache_dir(output_dir: Path, custom_cache: Path | None) -> Path:
    return (custom_cache or output_dir / "kernel-cache").resolve()


def parse_args() -> RunnerConfig:
    script_dir = Path(__file__).resolve().parent
    source_dir = script_dir.parents[2]
    default_build = source_dir / "build_cuda_stable"
    parser = argparse.ArgumentParser(
        description="Run a PoCL CUDA CTS suite with hard process-group timeouts"
    )
    parser.add_argument("--build-dir", type=Path, default=default_build)
    parser.add_argument("--install-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        help="kernel cache path; defaults inside the output directory",
    )
    suite_group = parser.add_mutually_exclusive_group()
    suite_group.add_argument("--suite", choices=KNOWN_SUITES, default=DEFAULT_SUITE)
    suite_group.add_argument(
        "--csv", type=Path, help="explicit CTS CSV manifest; overrides named suites"
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=DEFAULT_CTS_TIMEOUT_SECONDS,
        help=(
            "per-suite watchdog in seconds; 0 disables it "
            f"(default: {DEFAULT_CTS_TIMEOUT_SECONDS})"
        ),
    )
    parser.add_argument("--filter", dest="name_filter")
    args = parser.parse_args()
    build_dir = args.build_dir.resolve()
    install_dir = (args.install_dir or build_dir / "install").resolve()
    suite_name = args.csv.stem if args.csv is not None else args.suite
    csv_path = suite_csv_path(
        script_dir, build_dir, args.suite, custom_csv=args.csv
    )
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = (
        args.output_dir or build_dir / f"validation-logs/cts-{suite_name}-{timestamp}"
    ).resolve()
    cache_dir = selected_cache_dir(output_dir, args.cache_dir)
    if args.timeout < 0:
        parser.error("--timeout must be non-negative")
    timeout_seconds = args.timeout or None
    return RunnerConfig(
        build_dir=build_dir,
        install_dir=install_dir,
        output_dir=output_dir,
        timeout_seconds=timeout_seconds,
        name_filter=args.name_filter,
        suite_name=suite_name,
        csv_path=csv_path,
        cache_dir=cache_dir,
    )


def load_tests(config: RunnerConfig) -> tuple[TestCase, ...]:
    tests: list[TestCase] = []
    if not config.csv_path.is_file():
        raise RuntimeError(f"CTS CSV manifest does not exist: {config.csv_path}")
    with config.csv_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.reader(handle):
            if not row or row[0].lstrip().startswith("#"):
                continue
            if len(row) == 3:
                if row[0].strip() != "CL_DEVICE_TYPE_GPU":
                    continue
                name, command = row[1].strip(), row[2].strip()
            elif len(row) == 2:
                name, command = row[0].strip(), row[1].strip()
            else:
                raise RuntimeError(f"Invalid CTS CSV row: {row}")
            if config.name_filter and config.name_filter.lower() not in name.lower():
                continue
            tokens = shlex.split(command)
            executable = (config.cts_dir / tokens[0]).resolve()
            tests.append(TestCase(name, executable, tuple(tokens[1:])))
    if not tests:
        raise RuntimeError("No CTS tests selected")
    return tuple(tests)


def runtime_environment(config: RunnerConfig) -> dict[str, str]:
    environment = dict(os.environ)
    old_library_path = environment.get("LD_LIBRARY_PATH")
    library_path = str(config.install_dir / "lib")
    environment["LD_LIBRARY_PATH"] = (
        f"{library_path}:{old_library_path}" if old_library_path else library_path
    )
    environment["POCL_DEVICES"] = "cuda"
    environment["POCL_CACHE_DIR"] = str(config.cache_dir)
    environment.pop("POCL_CUDA_GPU_ARCH", None)
    return environment


def run_capture(command: tuple[str, ...], environment: dict[str, str]) -> str:
    completed = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
        env=environment,
    )
    return completed.stdout + completed.stderr


def resolved_opencl_library(executable: Path, environment: dict[str, str]) -> Path:
    output = run_capture(("ldd", str(executable)), environment)
    match = re.search(r"libOpenCL\.so\.\d+\s+=>\s+(\S+)", output)
    if match is None:
        raise RuntimeError(f"Unable to resolve libOpenCL for {executable}\n{output}")
    return Path(match.group(1)).resolve(strict=True)


def verify_inputs(
    config: RunnerConfig, tests: tuple[TestCase, ...], environment: dict[str, str]
) -> None:
    expected_library = config.library.resolve(strict=True)
    missing = [str(test.executable) for test in tests if not test.executable.is_file()]
    if missing:
        raise RuntimeError("Missing CTS executables:\n" + "\n".join(missing))
    for executable in sorted({test.executable for test in tests}):
        resolved = resolved_opencl_library(executable, environment)
        if resolved != expected_library:
            raise RuntimeError(
                f"{executable} resolves OpenCL to {resolved}, expected {expected_library}"
            )


def verify_device_contract(
    config: RunnerConfig, environment: dict[str, str]
) -> dict[str, object]:
    report_path = config.output_dir / "device-contract.json"
    verifier = Path(__file__).resolve().parent / "verify_device.py"
    run_capture(
        (
            sys.executable,
            str(verifier),
            "--library",
            str(config.library),
            "--output",
            str(report_path),
        ),
        environment,
    )
    return json.loads(report_path.read_text(encoding="utf-8"))


def gpu_processes() -> tuple[str, ...]:
    output = run_capture(
        (
            "nvidia-smi",
            "--query-compute-apps=pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ),
        dict(os.environ),
    )
    return tuple(line for line in output.splitlines() if line.strip())


def wait_for_gpu_idle() -> tuple[str, ...]:
    deadline = time.monotonic() + GPU_IDLE_WAIT_SECONDS
    processes = gpu_processes()
    while processes and time.monotonic() < deadline:
        time.sleep(GPU_IDLE_POLL_SECONDS)
        processes = gpu_processes()
    return processes


def record_environment(
    config: RunnerConfig,
    environment: dict[str, str],
    device_report: dict[str, object],
) -> None:
    source_dir = Path(__file__).resolve().parents[3]
    cts_source = config.build_dir / "examples/conformance/src/conformance"
    compute_info = config.cts_dir / "computeinfo/test_computeinfo"
    commands = {
        "gpu": (
            "nvidia-smi",
            "--query-gpu=name,driver_version,compute_cap,uuid",
            "--format=csv,noheader",
        ),
        "llvm": ("/usr/bin/llvm-config-18", "--version"),
        "pocl_commit": ("git", "-C", str(source_dir), "rev-parse", "HEAD"),
        "pocl_status": ("git", "-C", str(source_dir), "status", "--short"),
        "cts_commit": ("git", "-C", str(cts_source), "rev-parse", "HEAD"),
        "loader": ("ldd", str(compute_info)),
    }
    report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "suite": config.suite_name,
        "csv": str(config.csv_path),
        "cache_dir": str(config.cache_dir),
        "library": str(config.library.resolve(strict=True)),
        "build_dir": str(config.build_dir),
        "install_dir": str(config.install_dir),
        "timeout_seconds": config.timeout_seconds,
        "commands": {
            name: run_capture(command, environment).strip()
            for name, command in commands.items()
        },
        "device": device_report,
    }
    path = config.output_dir / "environment.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def terminate_process_group(process: subprocess.Popen[str]) -> None:
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=TERMINATION_GRACE_SECONDS)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()


def marker_lines(output: str) -> tuple[str, ...]:
    return tuple(line for line in output.splitlines() if FAILURE_PATTERN.search(line))


def suite_was_skipped(output: str) -> bool:
    return any(pattern.search(output) for pattern in SUITE_SKIP_PATTERNS)


def classify_result(
    *,
    timed_out: bool,
    crash_signal: int | None,
    return_code: int | None,
    failure_markers: tuple[str, ...],
    suite_skipped: bool,
    all_selected_not_supported: bool,
) -> str:
    if timed_out:
        return "timeout"
    if crash_signal is not None:
        return "crash"
    if return_code != 0 or failure_markers:
        return "fail"
    if suite_skipped or all_selected_not_supported:
        return "skip"
    return "pass"


def run_test(
    test: TestCase, config: RunnerConfig, environment: dict[str, str]
) -> TestResult:
    command = (str(test.executable), *test.arguments)
    start = time.monotonic()
    process = subprocess.Popen(
        command,
        cwd=test.executable.parent,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    timed_out = False
    try:
        stdout, stderr = process.communicate(timeout=config.timeout_seconds)
    except subprocess.TimeoutExpired:
        timed_out = True
        terminate_process_group(process)
        stdout, stderr = process.communicate()
    duration = time.monotonic() - start
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "-", test.name).strip("-").lower()
    stdout_path = config.output_dir / f"{slug}.stdout.log"
    stderr_path = config.output_dir / f"{slug}.stderr.log"
    stdout_path.write_text(stdout, encoding="utf-8")
    stderr_path.write_text(stderr, encoding="utf-8")
    combined_output = stdout + "\n" + stderr
    markers = marker_lines(combined_output)
    assessment = assess_output(combined_output)
    return_code = process.returncode
    crash_signal = -return_code if return_code is not None and return_code < 0 else None
    status = classify_result(
        timed_out=timed_out,
        crash_signal=crash_signal,
        return_code=return_code,
        failure_markers=markers,
        suite_skipped=suite_was_skipped(combined_output),
        all_selected_not_supported=assessment.all_selected_not_supported,
    )
    return TestResult(
        name=test.name,
        command=command,
        status=status,
        return_code=return_code,
        signal=crash_signal,
        duration_seconds=duration,
        timed_out=timed_out,
        failure_markers=markers,
        selected_test_count=assessment.selected_test_count,
        applicable_pass_count=assessment.applicable_pass_count,
        not_supported_count=assessment.not_supported_count,
        not_supported_tests=assessment.not_supported_tests,
        not_supported_reasons=assessment.not_supported_reasons,
        stdout_log=str(stdout_path),
        stderr_log=str(stderr_path),
    )


def write_summary(config: RunnerConfig, results: list[TestResult]) -> None:
    counts = {status: 0 for status in ("pass", "skip", "fail", "crash", "timeout")}
    for result in results:
        counts[result.status] += 1
    report = {"counts": counts, "results": [asdict(result) for result in results]}
    path = config.output_dir / "summary.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    config = parse_args()
    config.output_dir.mkdir(parents=True, exist_ok=False)
    environment = runtime_environment(config)
    tests = load_tests(config)
    initial_processes = wait_for_gpu_idle()
    if initial_processes:
        raise RuntimeError("GPU is not exclusive:\n" + "\n".join(initial_processes))
    verify_inputs(config, tests, environment)
    device_report = verify_device_contract(config, environment)
    validation_processes = wait_for_gpu_idle()
    if validation_processes:
        raise RuntimeError(
            "Device validation left GPU processes:\n" + "\n".join(validation_processes)
        )
    record_environment(config, environment, device_report)
    results: list[TestResult] = []
    for index, test in enumerate(tests, start=1):
        result = run_test(test, config, environment)
        results.append(result)
        write_summary(config, results)
        print(
            f"[{index:02d}/{len(tests):02d}] {result.status.upper():7s} "
            f"{result.name} ({result.duration_seconds:.1f}s)",
            flush=True,
        )
    remaining_processes = wait_for_gpu_idle()
    if remaining_processes:
        raise RuntimeError("GPU processes remain after CTS:\n" + "\n".join(remaining_processes))
    return 1 if any(result.status in {"fail", "crash", "timeout"} for result in results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
