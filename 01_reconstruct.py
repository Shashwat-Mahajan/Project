"""
Step 1: COLMAP reconstruction pipeline
=======================================
Runs feature extraction -> matching -> sparse mapping -> dense reconstruction
-> meshing. Produces everything needed for the later autonomous scale
calibration and pothole measurement steps.

Run this first:
    python 01_reconstruct.py
"""

import os
import subprocess

# -----------------------------------------------------------------
# CONFIG - edit these paths for your machine
# -----------------------------------------------------------------
COLMAP_DIR = r"C:\Users\ASUS\Downloads\colmap-x64-windows-cuda\bin"
PROJECT_DIR = r"D:\MinProject"

COLMAP_PATH = os.path.join(COLMAP_DIR, "colmap.exe")
DB_PATH = os.path.join(PROJECT_DIR, "db.db")
IMAGE_PATH = os.path.join(PROJECT_DIR, "images")
SPARSE_PATH = os.path.join(PROJECT_DIR, "sparse")
DENSE_PATH = os.path.join(PROJECT_DIR, "dense")

env = os.environ.copy()
env["QT_QPA_PLATFORM"] = "offscreen"
env["QT_QPA_PLATFORM_PLUGIN_PATH"] = os.path.join(COLMAP_DIR, "platforms")


def run(args, step_name):
    print(f"\n{'='*60}\n[STEP] {step_name}\n{'='*60}")
    subprocess.run([COLMAP_PATH] + args, check=True, cwd=COLMAP_DIR, env=env)


def main():
    os.makedirs(SPARSE_PATH, exist_ok=True)
    os.makedirs(DENSE_PATH, exist_ok=True)

    # 1. Feature extraction
    run(
        [
            "feature_extractor",
            "--database_path",
            DB_PATH,
            "--image_path",
            IMAGE_PATH,
            "--ImageReader.single_camera",
            "1",
        ],
        "Feature extraction",
    )

    # 2. Matching
    run(
        [
            "sequential_matcher",
            "--database_path",
            DB_PATH,
        ],
        "Sequential matching",
    )

    # 3. Sparse reconstruction
    run(
        [
            "mapper",
            "--database_path",
            DB_PATH,
            "--image_path",
            IMAGE_PATH,
            "--output_path",
            SPARSE_PATH,
        ],
        "Sparse mapping",
    )

    # 4. Undistort images for dense stage
    run(
        [
            "image_undistorter",
            "--image_path",
            IMAGE_PATH,
            "--input_path",
            os.path.join(SPARSE_PATH, "0"),
            "--output_path",
            DENSE_PATH,
            "--output_type",
            "COLMAP",
        ],
        "Image undistortion",
    )

    # 5. Dense stereo
    run(
        [
            "patch_match_stereo",
            "--workspace_path",
            DENSE_PATH,
            "--workspace_format",
            "COLMAP",
            "--PatchMatchStereo.max_image_size",
            "1200",
            "--PatchMatchStereo.num_iterations",
            "5",
        ],
        "Patch match stereo",
    )

    # 6. Fusion
    run(
        [
            "stereo_fusion",
            "--workspace_path",
            DENSE_PATH,
            "--workspace_format",
            "COLMAP",
            "--output_path",
            os.path.join(DENSE_PATH, "fused.ply"),
        ],
        "Stereo fusion",
    )

    # 7. Meshing (optional, not used downstream but kept for visualization)
    run(
        [
            "poisson_mesher",
            "--input_path",
            os.path.join(DENSE_PATH, "fused.ply"),
            "--output_path",
            os.path.join(DENSE_PATH, "meshed.ply"),
        ],
        "Poisson meshing",
    )

    print("\n✅ Reconstruction complete.")
    print(f"   Sparse model: {os.path.join(SPARSE_PATH, '0')}")
    print(f"   Dense (undistorted) sparse: {os.path.join(DENSE_PATH, 'sparse')}")
    print(f"   Dense point cloud: {os.path.join(DENSE_PATH, 'fused.ply')}")


if __name__ == "__main__":
    main()
