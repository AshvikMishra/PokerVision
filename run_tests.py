#!/usr/bin/env python3
"""Run every test script and report.

    python run_tests.py            # everything
    python run_tests.py gui detect # just the named ones
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
TESTS = ROOT / "tests"

ALL = ["capture", "cli", "gui", "ink", "detect", "recognition", "reading"]


def main(argv: list[str]) -> int:
    wanted = argv or ALL
    unknown = [w for w in wanted if w not in ALL]
    if unknown:
        print(f"unknown test(s): {', '.join(unknown)}\navailable: "
              f"{', '.join(ALL)}")
        return 2

    results: list[tuple[str, bool, float, str]] = []
    for name in wanted:
        script = TESTS / f"test_{name}.py"
        t0 = time.perf_counter()
        proc = subprocess.run([sys.executable, str(script)],
                              capture_output=True, text=True, cwd=ROOT)
        dt = time.perf_counter() - t0
        tail = [ln for ln in (proc.stdout or "").splitlines() if ln.strip()]
        verdict = tail[-1] if tail else "(no output)"
        results.append((name, proc.returncode == 0, dt, verdict))

    width = max(len(n) for n, _, _, _ in results)
    print()
    for name, ok, dt, verdict in results:
        print(f"  {name:<{width}}  {'pass' if ok else 'FAIL':<4} "
              f"{dt:6.1f}s   {verdict[:70]}")
    failed = [n for n, ok, _, _ in results if not ok]
    total = sum(dt for _, _, dt, _ in results)
    print(f"\n  {len(results) - len(failed)}/{len(results)} passed "
          f"in {total:.1f}s")
    if failed:
        print(f"  failed: {', '.join(failed)}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
