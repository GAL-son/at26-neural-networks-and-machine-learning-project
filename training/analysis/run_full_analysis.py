"""Run every dataset-analysis step in order: inventory, contact sheets, analyze, temporal.

    python training/analysis/run_full_analysis.py                        # full run, results in ./results
    python training/analysis/run_full_analysis.py --quick                # few frames per step: pipeline check in minutes
    python training/analysis/run_full_analysis.py --only analyze temporal

Interpret the results with training/analysis/ANALYSIS_GUIDE.md.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]

# (step, script, extra arguments in --quick mode)
STEPS = [
    ("inventory", "inventory.py", ["--stride", "10"]),
    ("contact_sheet", "contact_sheet.py", ["--frames", "1"]),
    ("analyze", "analyze.py", ["--frames", "8"]),
    ("temporal", "temporal.py", ["--pairs", "8", "--stride", "12", "--max-lag", "120"]),
]


def run_step(step: str, script: str, extra: list[str], captures: Path, out: Path, log_dir: Path) -> tuple[int, float]:
    command = [sys.executable, str(HERE / script), "--captures", str(captures), "--out", str(out), *extra]
    print(f"\n=== {step} ({script}{' ' + ' '.join(extra) if extra else ''})", flush=True)
    env = {**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8"}
    start = time.perf_counter()
    with open(log_dir / f"{step}.log", "wb") as log:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env, cwd=ROOT)
        for chunk in iter(lambda: process.stdout.read1(4096), b""):  # raw bytes keep the \r progress lines intact
            sys.stdout.buffer.write(chunk)
            sys.stdout.flush()
            log.write(chunk)
        code = process.wait()
    return code, time.perf_counter() - start


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--captures", type=Path, default=ROOT / "data" / "captures")
    parser.add_argument("--out", type=Path, default=ROOT / "results")
    parser.add_argument("--quick", action="store_true", help="few frames per step: checks the pipeline, numbers are rough")
    parser.add_argument("--only", nargs="+", choices=[step for step, _, _ in STEPS], help="run only these steps")
    args = parser.parse_args()

    captures, out = args.captures.resolve(), args.out.resolve()
    log_dir = out / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    summary = []
    for step, script, quick_args in STEPS:
        if args.only and step not in args.only:
            continue
        # steps are independent, so a failure is reported but does not stop the rest
        code, seconds = run_step(step, script, quick_args if args.quick else [], captures, out, log_dir)
        summary.append((step, code, seconds))

    print("\n=== summary" + (" (QUICK MODE - numbers are rough)" if args.quick else ""))
    for step, code, seconds in summary:
        print(f"  {step:<14} {'ok' if code == 0 else f'FAILED (exit {code})':<18} {seconds / 60:5.1f} min")
    print(f"\nResults: {out}\nLogs:    {log_dir}\nGuide:   {HERE / 'ANALYSIS_GUIDE.md'}")

    if any(code != 0 for _, code, _ in summary):
        sys.exit(1)


if __name__ == "__main__":
    main()



