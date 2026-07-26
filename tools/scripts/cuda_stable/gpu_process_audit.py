#!/usr/bin/env python3

from __future__ import annotations

import json
import os
import subprocess
import time
from collections.abc import Callable
from pathlib import Path

GPU_IDLE_WAIT_SECONDS = 10
GPU_IDLE_POLL_SECONDS = 0.2

ProcessQuery = Callable[[], tuple[str, ...]]


def gpu_processes() -> tuple[str, ...]:
    completed = subprocess.run(
        (
            "nvidia-smi",
            "--query-compute-apps=pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ),
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
        env=dict(os.environ),
    )
    return tuple(line for line in completed.stdout.splitlines() if line.strip())


def wait_for_gpu_idle(process_query: ProcessQuery = gpu_processes) -> tuple[str, ...]:
    deadline = time.monotonic() + GPU_IDLE_WAIT_SECONDS
    processes = process_query()
    while processes and time.monotonic() < deadline:
        time.sleep(GPU_IDLE_POLL_SECONDS)
        processes = process_query()
    return processes


def gpu_process_snapshot(
    allow_shared_gpu: bool, process_query: ProcessQuery = gpu_processes
) -> tuple[str, ...]:
    return process_query() if allow_shared_gpu else wait_for_gpu_idle(process_query)


def enforce_gpu_process_policy(
    processes: tuple[str, ...], *, allow_shared_gpu: bool, error_message: str
) -> None:
    if processes and not allow_shared_gpu:
        raise RuntimeError(error_message + ":\n" + "\n".join(processes))


def write_gpu_process_report(
    output_dir: Path,
    *,
    allow_shared_gpu: bool,
    initial: tuple[str, ...],
    after_validation: tuple[str, ...] | None = None,
    final: tuple[str, ...] | None = None,
) -> None:
    report: dict[str, object] = {
        "allow_shared_gpu": allow_shared_gpu,
        "initial": list(initial),
    }
    if after_validation is not None:
        report["after_device_validation"] = list(after_validation)
    if final is not None:
        report["final"] = list(final)
    path = output_dir / "gpu-processes.json"
    path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
