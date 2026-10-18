#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Merge-gate checks for the matmul/fused/reduce family.

count    Fail if ``pytest -m merge_gate`` under those dirs does not match
         tests/pipeline_reorg/ttnn_matmul_fused_reduction_ops.count.
changed  Run ``test_*.py`` files a PR adds or edits under those dirs, including
         unmarked tests. No-op when the diff touches none of them.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

FAMILY_DIRS = (
    "tests/ttnn/unit_tests/operations/matmul",
    "tests/ttnn/unit_tests/operations/fused",
    "tests/ttnn/unit_tests/operations/reduce",
)
COUNT_FILE = Path("tests/pipeline_reorg/ttnn_matmul_fused_reduction_ops.count")
_COLLECTED_RE = re.compile(r"(\d+)(?:/\d+)? tests collected")


def _repo() -> Path:
    return Path(__file__).resolve().parents[2]


def _run(cmd: list[str], cwd: Path, check: bool = False) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, check=check)


def _expected_count(text: str) -> int:
    for raw in text.splitlines():
        line = raw.strip()
        if line and not line.startswith("#"):
            return int(line.split()[0])
    raise ValueError("count file has no integer")


def _collected_count(output: str) -> int:
    matches = _COLLECTED_RE.findall(output)
    if not matches:
        raise ValueError(f"no 'N tests collected' summary in pytest output:\n{output[-500:]}")
    return int(matches[-1])


def check_count() -> int:
    repo = _repo()
    expected = _expected_count((repo / COUNT_FILE).read_text())
    result = _run(
        [
            sys.executable,
            "-m",
            "pytest",
            "--collect-only",
            "-q",
            "-o",
            "addopts=",
            "-m",
            "merge_gate",
            *FAMILY_DIRS,
        ],
        cwd=repo,
    )
    output = result.stdout + result.stderr
    if result.returncode != 0:
        sys.stderr.write(output + "collection-count guard: pytest --collect-only failed\n")
        return result.returncode
    collected = _collected_count(output)
    if collected != expected:
        sys.stderr.write(
            f"collection-count guard: collected {collected} merge_gate tests, expected {expected} "
            f"(from {COUNT_FILE}). Update the count file when you add or remove marks.\n"
        )
        return 1
    print(f"collection-count guard: {collected} merge_gate tests (matches {COUNT_FILE})")
    return 0


def _have_commit(rev: str, cwd: Path) -> bool:
    return _run(["git", "cat-file", "-e", f"{rev}^{{commit}}"], cwd=cwd).returncode == 0


def _resolve_base(explicit: str | None) -> str:
    if explicit:
        return explicit
    for key in ("GITHUB_EVENT_PULL_REQUEST_BASE_SHA", "GITHUB_BASE_SHA"):
        value = os.environ.get(key)
        if value:
            return value
    base_ref = os.environ.get("GITHUB_BASE_REF") or "main"
    return base_ref if "/" in base_ref else f"origin/{base_ref}"


def _ensure_revision(rev: str, cwd: Path) -> str:
    if _have_commit(rev, cwd):
        return rev
    fetch_ref = rev.removeprefix("origin/")
    _run(["git", "fetch", "--depth=1", "origin", fetch_ref], cwd=cwd, check=True)
    if _have_commit(rev, cwd):
        return rev
    if _have_commit("FETCH_HEAD", cwd):
        return "FETCH_HEAD"
    raise RuntimeError(f"could not resolve diff base {rev!r} after fetching origin {fetch_ref!r}")


def _is_family_test(path: str) -> bool:
    name = Path(path).name
    if not path.endswith(".py") or not name.startswith("test_"):
        return False
    return any(path == directory or path.startswith(directory + "/") for directory in FAMILY_DIRS)


def _changed_test_files(base: str, cwd: Path) -> list[str]:
    result = _run(
        ["git", "diff", "--name-only", "--diff-filter=AMR", "-M", f"{base}...HEAD"],
        cwd=cwd,
        check=True,
    )
    return [path for path in result.stdout.splitlines() if _is_family_test(path)]


def run_changed(explicit_base: str | None, dry_run: bool) -> int:
    repo = _repo()
    base = _ensure_revision(_resolve_base(explicit_base), repo)
    targets = _changed_test_files(base, repo)
    if not targets:
        print(f"changed-test gate: no matmul/fused/reduce test files changed vs {base}")
        return 0
    print("changed-test gate: running")
    for target in targets:
        print(f"  {target}")
    if dry_run:
        return 0
    return subprocess.run(
        [sys.executable, "-m", "pytest", "--timeout", "60", "-xv", *targets],
        cwd=repo,
        check=False,
    ).returncode


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("count", "changed"))
    parser.add_argument("--base", default=None, help="git revision to diff against (changed only)")
    parser.add_argument("--dry-run", action="store_true", help="print selected files and exit (changed only)")
    args = parser.parse_args()
    if args.command == "count":
        return check_count()
    return run_changed(args.base, args.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())
