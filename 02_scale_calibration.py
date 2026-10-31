"""
Step 2: Fully autonomous scale calibration
============================================
Detects the ArUco marker's 4 corners in 2D in every image (sub-pixel
accurate), then triangulates each corner's 3D position using COLMAP's
own camera poses (from the DENSE/undistorted sparse model, so the
coordinate frame matches dense/fused.ply exactly). No manual clicking.

SETUP (must be done during capture, before running this):
  1. Print an ArUco marker (DICT_4X4_50, ID 0) at a KNOWN physical size,
     e.g. exactly 10.0 cm x 10.0 cm side length. Print flat, no folds.
     Free generator: https://chev.me/arucogen/
  2. Place it flat on the road next to the pothole, visible in several
     photos fed into COLMAP.
  3. Measure the printed marker's actual black-square side length with
     a ruler and set MARKER_SIZE_CM below.

Requires: pip install pycolmap opencv-contrib-python

Run after 01_reconstruct.py:
    python 02_scale_calibration.py
Prints and saves scale_factor.json for the next step to consume.
"""

import os
import glob
import json
import cv2
import numpy as np

# -----------------------------------------------------------------
# CONFIG
# -----------------------------------------------------------------
PROJECT_DIR = r"D:\MinProject"
IMAGE_DIR = os.path.join(PROJECT_DIR, "images")
# Use the UNDISTORTED sparse model (matches dense/fused.ply's coordinate frame)
DENSE_SPARSE_PATH = os.path.join(PROJECT_DIR, "dense", "sparse")
OUTPUT_JSON = os.path.join(PROJECT_DIR, "scale_factor.json")

MARKER_SIZE_CM = 10.0
ARUCO_DICT = cv2.aruco.DICT_4X4_50
MIN_VIEWS_REQUIRED = 2  # minimum images marker must be seen in per corner


# -----------------------------------------------------------------
# STEP A: Detect marker corners in 2D across all images
# -----------------------------------------------------------------
def detect_marker_corners_2d(image_dir):
    aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
    aruco_params = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)

    image_paths = sorted(
        glob.glob(os.path.join(image_dir, "*.jpg"))
        + glob.glob(os.path.join(image_dir, "*.JPG"))
        + glob.glob(os.path.join(image_dir, "*.jpeg"))
        + glob.glob(os.path.join(image_dir, "*.JPEG"))
        + glob.glob(os.path.join(image_dir, "*.png"))
        + glob.glob(os.path.join(image_dir, "*.PNG"))
    )
    print(f"Found {len(image_paths)} images in {image_dir}")

    detections = {}
    for path in image_paths:
        img = cv2.imread(path)
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = detector.detectMarkers(gray)
        if ids is not None and len(ids) > 0:
            detections[os.path.basename(path)] = corners[0][0]  # 4x2 pixel coords

    print(f"Marker detected in {len(detections)} / {len(image_paths)} images")
    return detections


# -----------------------------------------------------------------
# STEP B: Triangulate each corner's 3D position from COLMAP poses
# -----------------------------------------------------------------
def triangulate_marker_corners(sparse_path, detections):
    import pycolmap

    reconstruction = pycolmap.Reconstruction(sparse_path)
    name_to_image = {img.name: img for img in reconstruction.images.values()}
    cameras = reconstruction.cameras

    corner_rays = [[] for _ in range(4)]
    used_images = 0

    for img_name, corners_2d in detections.items():
        if img_name not in name_to_image:
            continue
        image = name_to_image[img_name]
        camera = cameras[image.camera_id]
        cam_from_world = image.cam_from_world
        world_from_cam = cam_from_world.inverse()
        cam_center = world_from_cam.translation
        used_images += 1

        for i, px in enumerate(corners_2d):
            ray_cam = camera.cam_from_img(px)
            ray_cam = np.array([ray_cam[0], ray_cam[1], 1.0])
            ray_world = world_from_cam.rotation * ray_cam
            ray_world = ray_world / np.linalg.norm(ray_world)
            corner_rays[i].append((cam_center, ray_world))

    print(f"Marker images matched to COLMAP poses: {used_images}")

    def triangulate_rays(rays):
        # Least-squares closest point to a bundle of 3D lines
        A = np.zeros((3, 3))
        b = np.zeros(3)
        for origin, d in rays:
            d = np.asarray(d).reshape(3, 1)
            proj = np.eye(3) - d @ d.T
            A += proj
            b += proj @ np.asarray(origin)
        return np.linalg.solve(A, b)

    corners_3d = []
    for i, rays in enumerate(corner_rays):
        if len(rays) < MIN_VIEWS_REQUIRED:
            raise RuntimeError(
                f"Corner {i} only seen in {len(rays)} image(s); "
                f"need at least {MIN_VIEWS_REQUIRED}. Move the marker more "
                f"centrally in the scene or capture more overlapping photos."
            )
        corners_3d.append(triangulate_rays(rays))

    return np.array(corners_3d)


# -----------------------------------------------------------------
# STEP C: Compute scale factor + a reprojection-based sanity score
# -----------------------------------------------------------------
def compute_scale_factor(corners_3d):
    top = np.linalg.norm(corners_3d[0] - corners_3d[1])
    right = np.linalg.norm(corners_3d[1] - corners_3d[2])
    bottom = np.linalg.norm(corners_3d[2] - corners_3d[3])
    left = np.linalg.norm(corners_3d[3] - corners_3d[0])

    side_lengths = [top, right, bottom, left]
    measured_units = float(np.mean(side_lengths))
    side_std = float(np.std(side_lengths))

    scale_factor = MARKER_SIZE_CM / measured_units
    # relative disagreement between the 4 sides = confidence signal
    disagreement_pct = side_std / measured_units if measured_units > 0 else 1.0

    return scale_factor, measured_units, disagreement_pct


def main():
    detections = detect_marker_corners_2d(IMAGE_DIR)
    if len(detections) < MIN_VIEWS_REQUIRED:
        raise SystemExit(
            f"Marker only detected in {len(detections)} image(s). "
            f"Need at least {MIN_VIEWS_REQUIRED}. Check marker visibility, "
            f"lighting, and focus."
        )

    corners_3d = triangulate_marker_corners(DENSE_SPARSE_PATH, detections)
    scale_factor, measured_units, disagreement_pct = compute_scale_factor(corners_3d)

    print(f"\nTriangulated marker side (avg of 4 edges): {measured_units:.5f} units")
    print(f"Side-length disagreement: {disagreement_pct*100:.2f}%")
    print(f"Scale factor: {scale_factor:.4f} cm/unit")

    if disagreement_pct > 0.05:
        print(
            "\nWARNING: the 4 triangulated sides disagree by >5%. This usually "
            "means the marker was flexed/not flat, poorly detected in some "
            "images, or the sparse/dense coordinate frames don't match. "
            "Treat this scale factor with caution."
        )

    with open(OUTPUT_JSON, "w") as f:
        json.dump(
            {
                "scale_factor_cm_per_unit": scale_factor,
                "measured_marker_units": measured_units,
                "disagreement_pct": disagreement_pct,
                "marker_size_cm": MARKER_SIZE_CM,
                "num_images_used": len(detections),
            },
            f,
            indent=2,
        )

    print(f"\n✅ Saved scale factor to {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
