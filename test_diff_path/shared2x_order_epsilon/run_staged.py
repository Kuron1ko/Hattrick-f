from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


THIS_DIR = Path(__file__).resolve().parent
PYTHON = sys.executable


def run(*arguments: str, allow_gate_failure: bool = False) -> bool:
    result = subprocess.run(
        [PYTHON, "-B", str(THIS_DIR / "run_experiment.py"), *arguments],
        cwd=THIS_DIR,
        check=False,
    )
    if result.returncode and not allow_gate_failure:
        raise SystemExit(result.returncode)
    return result.returncode == 0


def main() -> None:
    run("--level", "0", "--approach", "control", "--seed", "490")
    run("--level", "1", "--approach", "control", "--seed", "490")
    run("--level", "1", "--approach", "swap", "--seed", "490")
    run(
        "--level", "1", "--approach", "epsilon", "--epsilon", "0.005", "--seed", "490"
    )
    subprocess.run([PYTHON, "-B", str(THIS_DIR / "integration_checks.py")], cwd=THIS_DIR, check=True)
    phase_a_ok = run(
        "--level", "2", "--approach", "control", "--seed", "490", allow_gate_failure=True
    )
    if not phase_a_ok:
        subprocess.run([PYTHON, "-B", str(THIS_DIR / "analyze_results.py")], cwd=THIS_DIR, check=True)
        return
    for approach, epsilon in (
        ("swap", None),
        ("epsilon", "0.0025"),
        ("epsilon", "0.005"),
        ("epsilon", "0.01"),
    ):
        arguments = ["--level", "2", "--approach", approach, "--seed", "490"]
        if epsilon is not None:
            arguments += ["--epsilon", epsilon]
        run(*arguments)
    subprocess.run([PYTHON, "-B", str(THIS_DIR / "analyze_results.py")], cwd=THIS_DIR, check=True)


if __name__ == "__main__":
    main()
