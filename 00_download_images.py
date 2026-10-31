"""
Full autonomous pipeline: reconstruction -> scale calibration -> measurement
================================================================================
Runs all three steps back to back with no manual interaction required,
provided the ArUco marker was placed correctly during capture.

Usage:
    python run_pipeline.py                # single (largest) pothole
    python run_pipeline.py --all-clusters # measure every detected pothole
"""

import subprocess
import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

STEPS = [
    ("00_download_images.py", []),
    ("01_reconstruct.py", []),
    ("02_scale_calibration.py", []),
    (
        "03_measure_pothole.py",
        ["--all-clusters"] if "--all-clusters" in sys.argv else [],
    ),
]


def main():
    for script, extra_args in STEPS:
        path = os.path.join(SCRIPT_DIR, script)
        print(f"\n{'#'*70}\n# RUNNING: {script}\n{'#'*70}")
        result = subprocess.run([sys.executable, path] + extra_args)
        if result.returncode != 0:
            print(
                f"\n❌ Pipeline stopped: {script} exited with code {result.returncode}"
            )
            sys.exit(result.returncode)

    print("\n\n✅ FULL AUTONOMOUS PIPELINE COMPLETE")
    print("   Check D:\\MinProject\\pothole_measurements.json for results.")


if __name__ == "__main__":
    main()
