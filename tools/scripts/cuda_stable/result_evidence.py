#!/usr/bin/env python3

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any


def write_summary(output_dir: Path, results: Sequence[Any]) -> None:
    counts = {status: 0 for status in ("pass", "skip", "fail", "crash", "timeout")}
    for result in results:
        counts[result.status] += 1
    report = {"counts": counts, "results": [asdict(result) for result in results]}
    path = output_dir / "summary.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_checksums(output_dir: Path) -> None:
    checksums = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(output_dir.iterdir())
        if path.is_file() and path.name != "sha256sums.json"
    }
    path = output_dir / "sha256sums.json"
    path.write_text(json.dumps(checksums, indent=2, sort_keys=True) + "\n", encoding="utf-8")
