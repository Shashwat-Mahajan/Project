"""
Step 1 (patched): COLMAP reconstruction pipeline
================================================
feature extraction -> matching -> sparse mapping -> undistortion -> dense stereo
-> fusion -> (optional) meshing.

What changed vs the original, and why (found by reading a real run's log):
  * The mapper can write SEVERAL models (sparse/0, sparse/1, ...). The original always
    used sparse/0 - in your run that model had only 4 of 17 images, so the whole dense
    stage used 4 images. Now the model with the MOST registered images is used.
  * Exhaustive matching (best for a small set of photos around one spot).
  * Sanity checks after each stage: registered images, depth maps, dense point count.
  * --fresh deletes the old database/sparse/dense so stale files cannot sneak in.

Usage (from D:\\MinProject):
    python 01_reconstruct.py            # keep existing database / outputs
    python 01_reconstruct.py --fresh    # delete db.db, sparse\\, dense\\ first (use when photos change)
"""

import os, sys, shutil, subprocess

# ---------------------------------------------------------------- CONFIG
COLMAP_DIR = r"C:\Users\ASUS\Downloads\colmap-x64-windows-cuda\bin"
PROJECT_DIR = os.environ.get("POTHOLE_PROJECT_DIR", r"D:\MinProject")
COLMAP_PATH = os.path.join(COLMAP_DIR, "colmap.exe")
DB_PATH = os.path.join(PROJECT_DIR, "db.db")
IMAGE_PATH = os.path.join(PROJECT_DIR, "images")
SPARSE_PATH = os.path.join(PROJECT_DIR, "sparse")
DENSE_PATH = os.path.join(PROJECT_DIR, "dense")

PM_MAX_IMAGE_SIZE = (
    1200  # raise to 2000 if your GPU has enough memory (more detail = better depth)
)
MIN_DENSE_POINTS = (
    100_000  # below this the cloud is too sparse to measure a 1-3 cm depression
)

env = os.environ.copy()
env["QT_QPA_PLATFORM"] = "offscreen"
env["QT_QPA_PLATFORM_PLUGIN_PATH"] = os.path.join(COLMAP_DIR, "platforms")


# ---------------------------------------------------------------- helpers
def run(args, step_name, check=True):
    print(f"\n{'='*60}\n[STEP] {step_name}\n{'='*60}")
    return subprocess.run([COLMAP_PATH] + args, check=check, cwd=COLMAP_DIR, env=env)


def list_images(folder):
    """Each real image file once (works the same on Windows and Linux)."""
    return sorted(
        f for f in os.listdir(folder) if f.lower().endswith((".jpg", ".jpeg", ".png"))
    )


def registered_images(model_dir):
    import pycolmap

    n = pycolmap.Reconstruction(model_dir).num_reg_images
    return int(n() if callable(n) else n)


def pick_best_model(sparse_path):
    models = sorted(
        d
        for d in os.listdir(sparse_path)
        if d.isdigit() and os.path.isdir(os.path.join(sparse_path, d))
    )
    if not models:
        raise SystemExit(
            "Mapper produced no model. Photos need more overlap / texture."
        )
    counts = {m: registered_images(os.path.join(sparse_path, m)) for m in models}
    print("\nRegistered images per sparse model:", counts)
    best = max(counts, key=counts.get)
    return os.path.join(sparse_path, best), counts[best]


def ply_vertex_count(path):
    with open(path, "rb") as f:
        for line in f:
            line = line.decode("ascii", "ignore").strip()
            if line.startswith("element vertex"):
                return int(line.split()[-1])
            if line == "end_header":
                break
    return 0


def count_files(folder, suffix):
    return (
        len([f for f in os.listdir(folder) if f.endswith(suffix)])
        if os.path.isdir(folder)
        else 0
    )


def main():
    n_images = len(list_images(IMAGE_PATH))
    print(f"Found {n_images} images in {IMAGE_PATH}")

    if "--fresh" in sys.argv:
        for p in (DB_PATH, SPARSE_PATH, DENSE_PATH):
            if os.path.isdir(p):
                shutil.rmtree(p)
                print("deleted folder", p)
            elif os.path.isfile(p):
                os.remove(p)
                print("deleted file  ", p)
    os.makedirs(SPARSE_PATH, exist_ok=True)
    os.makedirs(DENSE_PATH, exist_ok=True)

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

    run(["exhaustive_matcher", "--database_path", DB_PATH], "Exhaustive matching")

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

    best_model, n_reg = pick_best_model(SPARSE_PATH)
    print(f"Using model {best_model}  ({n_reg} of {n_images} images registered)")
    if n_reg < 0.6 * n_images:
        print(
            "\n!! WARNING: fewer than 60% of the photos were registered. The dense result will be poor.\n"
            "   Re-shoot with more overlap (each photo should share ~70% with its neighbours),\n"
            "   steady light, no zoom, and a textured surface.\n"
        )

    run(
        [
            "image_undistorter",
            "--image_path",
            IMAGE_PATH,
            "--input_path",
            best_model,
            "--output_path",
            DENSE_PATH,
            "--output_type",
            "COLMAP",
        ],
        "Image undistortion",
    )

    run(
        [
            "patch_match_stereo",
            "--workspace_path",
            DENSE_PATH,
            "--workspace_format",
            "COLMAP",
            "--PatchMatchStereo.max_image_size",
            str(PM_MAX_IMAGE_SIZE),
            "--PatchMatchStereo.num_iterations",
            "5",
        ],
        "Patch match stereo",
    )

    n_depth = count_files(
        os.path.join(DENSE_PATH, "stereo", "depth_maps"), ".geometric.bin"
    )
    print(f"Geometric depth maps found: {n_depth} (expected {n_reg})")
    if n_depth < n_reg:
        print(
            "!! WARNING: fewer depth maps than registered images - patch match did not finish properly.\n"
            "   Re-run with --fresh. If it keeps happening, check GPU memory (lower PM_MAX_IMAGE_SIZE)."
        )

    fused = os.path.join(DENSE_PATH, "fused.ply")
    run(
        [
            "stereo_fusion",
            "--workspace_path",
            DENSE_PATH,
            "--workspace_format",
            "COLMAP",
            "--output_path",
            fused,
        ],
        "Stereo fusion",
    )

    n_pts = ply_vertex_count(fused) if os.path.isfile(fused) else 0
    print(f"Dense point cloud: {n_pts:,} points")
    if n_pts < MIN_DENSE_POINTS:
        print(
            f"!! WARNING: only {n_pts:,} points (need >= {MIN_DENSE_POINTS:,}). Too sparse to measure a shallow\n"
            "   pothole. Usual causes: few registered images, stale files (use --fresh), low texture."
        )

    run(
        [
            "poisson_mesher",
            "--input_path",
            fused,
            "--output_path",
            os.path.join(DENSE_PATH, "meshed.ply"),
        ],
        "Poisson meshing (optional)",
        check=False,
    )
    if not os.path.isfile(os.path.join(DENSE_PATH, "meshed.ply")):
        print("(No meshed.ply was written - fine, the measurement uses fused.ply.)")

    print("\nReconstruction finished.")
    print(f"  Model used            : {best_model}  ({n_reg}/{n_images} images)")
    print(f"  Dense (undistorted)   : {os.path.join(DENSE_PATH, 'sparse')}")
    print(f"  Dense point cloud     : {fused}  ({n_pts:,} points)")


if __name__ == "__main__":
    main()
