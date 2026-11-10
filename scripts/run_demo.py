"""Prepare, verify, and launch the HealDeal synthetic-data demonstration.

Run from any working directory with:
    python scripts/run_demo.py

The runner regenerates the deterministic synthetic dataset, trains the
experimental cost model, executes the full unittest suite, and starts Flask.
It does not connect to real hospitals, insurers, patients, or clinical systems.
"""
from __future__ import annotations

import argparse
import importlib.util
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

REQUIRED_MODULES = {
    "flask": "Flask",
    "flask_cors": "Flask-Cors",
    "dotenv": "python-dotenv",
    "requests": "requests",
    "pandas": "pandas",
    "numpy": "numpy",
    "sklearn": "scikit-learn",
    "geopy": "geopy",
}


def run_step(label: str, command: list[str]) -> None:
    print(f"\n{'=' * 68}\n{label}\n{'=' * 68}", flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the complete HealDeal demo workflow.")
    parser.add_argument("--skip-tests", action="store_true", help="Skip automated tests.")
    parser.add_argument("--no-server", action="store_true", help="Prepare and test without starting Flask.")
    args = parser.parse_args()

    missing = [
        package for module, package in REQUIRED_MODULES.items()
        if importlib.util.find_spec(module) is None
    ]
    if missing:
        print("Missing required dependencies: " + ", ".join(missing), file=sys.stderr)
        print("Install them with: python -m pip install -r requirements.txt", file=sys.stderr)
        return 1

    run_step("STEP 1/3 — Generate the reproducible synthetic dataset", [
        sys.executable, "data/generate_dataset.py",
    ])
    run_step("STEP 2/3 — Train and evaluate the experimental cost model", [
        sys.executable, "ml/cost_model.py",
    ])

    if not args.skip_tests:
        run_step("STEP 3/3 — Run the complete automated test suite", [
            sys.executable, "-m", "unittest", "discover", "-s", "tests",
            "-p", "test_*.py", "-v",
        ])

    if args.no_server:
        print("\nHealDeal preparation and verification finished. Flask was not started.")
        return 0

    print("\nStarting HealDeal at http://127.0.0.1:5000", flush=True)
    print("Press Ctrl+C to stop the server. Data shown by this demo is synthetic only.\n", flush=True)
    try:
        subprocess.run([sys.executable, "backend/app.py"], cwd=ROOT, check=False)
    except KeyboardInterrupt:
        print("\nHealDeal server stopped.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
