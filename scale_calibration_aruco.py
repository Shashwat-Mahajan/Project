"""
ArUco marker-based scale calibration
======================================
Replaces manual "measure paper width in the point cloud" with automatic,
sub-pixel-accurate scale detection.

SETUP (do this BEFORE/DURING capture):
  1. Print an ArUco marker (e.g. DICT_4X4_50, ID 0) at a KNOWN physical
     size, e.g. exactly 10.0 cm x 10.0 cm. Print it flat, no folds.
     Free generator: https://chev.me/arucogen/
  2. Place it flat on the road, right next to the pothole, visible in
     several of the photos you feed into COLMAP.
  3. Measure the printed marker's actual black-square side length with
     a ruler (printers aren't always exact) and set MARKER_SIZE_CM below.

HOW THIS WORKS:
  - Detects the marker's 4 corners in 2D in each image (sub-pixel accurate).
  - If you don't need the full 2D->3D correspondence pipeline, use the
    FALLBACK method below: pick the same 4 corners directly on the dense
    point cloud in Open3D. This is simpler, still far more accurate than
    eyeballing a paper edge, and is the recommended path for most setups.
"""

import cv2
import numpy as np
import glob
import os

# -----------------------------------------------------------------
# CONFIG
# -----------------------------------------------------------------
IMAGE_DIR = r"D:\MinProject\images"
MARKER_SIZE_CM = 10.0  # real physical size of the printed marker's side
ARUCO_DICT = cv2.aruco.DICT_4X4_50

# -----------------------------------------------------------------
# STEP 1: Detect marker corners in each image (2D, sub-pixel accurate)
# Useful as a sanity check that your marker is visible/detectable at all.
# -----------------------------------------------------------------
aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
aruco_params = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)

image_paths = sorted(
    glob.glob(os.path.join(IMAGE_DIR, "*.jpg"))
    + glob.glob(os.path.join(IMAGE_DIR, "*.JPG"))
    + glob.glob(os.path.join(IMAGE_DIR, "*.jpeg"))
    + glob.glob(os.path.join(IMAGE_DIR, "*.JPEG"))
    + glob.glob(os.path.join(IMAGE_DIR, "*.png"))
    + glob.glob(os.path.join(IMAGE_DIR, "*.PNG"))
)
print(f"Found {len(image_paths)} image files in {IMAGE_DIR}")

detections = {}  # image_name -> corners (4x2 array, pixel coords)

for path in image_paths:
    img = cv2.imread(path)
    if img is None:
        continue
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = detector.detectMarkers(gray)
    if ids is not None and len(ids) > 0:
        detections[os.path.basename(path)] = corners[0][0]  # first marker found

print(f"Marker detected in {len(detections)} / {len(image_paths)} images")

if len(detections) < 2:
    print(
        "Marker not detected in enough images. Check lighting/focus/"
        "marker visibility, or just use the FALLBACK method below."
    )
else:
    for name, c in list(detections.items())[:3]:
        print(f"  {name}: corners = {c.tolist()}")

# -----------------------------------------------------------------
# RECOMMENDED METHOD: pick the 4 marker corners directly in the DENSE
# point cloud using Open3D's interactive point picker. Run this as a
# separate step after your dense reconstruction finishes.
#
# Usage: python scale_calibration_aruco.py --pick
# -----------------------------------------------------------------
if "--pick" in os.sys.argv:
    import open3d as o3d

    pcd = o3d.io.read_point_cloud("dense/fused.ply")

    print("\nShift+click the 4 corners of your ArUco marker IN ORDER:")
    print("  1) top-left  2) top-right  3) bottom-right  4) bottom-left")
    print("Then close the window (press Q).")

    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window(width=1280, height=800)
    vis.add_geometry(pcd)
    vis.get_render_option().point_size = 3.0
    vis.run()
    vis.destroy_window()

    picked_indices = vis.get_picked_points()
    if len(picked_indices) < 4:
        print(f"Only {len(picked_indices)} points picked — need exactly 4. Re-run.")
    else:
        points = np.asarray(pcd.points)
        corners = points[picked_indices[:4]]

        # Average opposite side lengths for a more robust size estimate
        top = np.linalg.norm(corners[0] - corners[1])
        right = np.linalg.norm(corners[1] - corners[2])
        bottom = np.linalg.norm(corners[2] - corners[3])
        left = np.linalg.norm(corners[3] - corners[0])
        measured_units = (top + bottom + right + left) / 4  # average all 4 sides

        scale_factor = MARKER_SIZE_CM / measured_units

        print(f"\nMeasured marker side (avg of 4 edges): {measured_units:.5f} units")
        print(f"Scale factor: {scale_factor:.4f} cm/unit")
        print(f"\n--> Copy this scale_factor into your measurement script.")
