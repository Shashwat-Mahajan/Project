"""
Step 2 (patched): autonomous scale calibration from an ArUco marker
===================================================================
Detects the marker's 4 corners in every image, triangulates them with COLMAP's camera
poses and converts the model's arbitrary units to centimetres.

What changed vs the original, and why:
  * Images were counted TWICE on Windows (the glob for *.jpg and *.JPG both match the same
    file, so 17 photos showed up as 34). Now each file is listed once.
  * Detection now runs on the UNDISTORTED images (dense/images) because the camera poses
    come from the undistorted model - pixels and camera model now match.
  * Only marker ID 0 is used (before, whatever marker was found first was used).
  * Sub-pixel corner refinement is switched on (the docstring promised it, the code
    did not enable it).
  * If nothing is found, it tries other ArUco dictionaries and tells you which one matches.

Usage:  python 02_scale_calibration.py
"""

import os, json
import cv2
import numpy as np

PROJECT_DIR = os.environ.get("POTHOLE_PROJECT_DIR", r"D:\MinProject")
ORIGINAL_IMAGES = os.path.join(PROJECT_DIR, "images")
UNDISTORTED_IMGS = os.path.join(PROJECT_DIR, "dense", "images")
DENSE_SPARSE_PATH = os.path.join(PROJECT_DIR, "dense", "sparse")
OUTPUT_JSON = os.path.join(PROJECT_DIR, "scale_factor.json")

MARKER_SIZE_CM = 10.0  # MEASURE the printed black square with a ruler!
ARUCO_DICT_NAME = "DICT_4X4_50"
MARKER_ID = 0
MIN_VIEWS_REQUIRED = 2

DICT_CANDIDATES = [
    "DICT_4X4_50",
    "DICT_4X4_100",
    "DICT_4X4_250",
    "DICT_4X4_1000",
    "DICT_5X5_50",
    "DICT_5X5_100",
    "DICT_5X5_250",
    "DICT_5X5_1000",
    "DICT_6X6_50",
    "DICT_6X6_100",
    "DICT_6X6_250",
    "DICT_6X6_1000",
    "DICT_7X7_50",
    "DICT_7X7_100",
    "DICT_7X7_250",
    "DICT_7X7_1000",
    "DICT_ARUCO_ORIGINAL",
    "DICT_APRILTAG_16h5",
    "DICT_APRILTAG_25h9",
    "DICT_APRILTAG_36h10",
    "DICT_APRILTAG_36h11",
]


def list_images(folder):
    return sorted(
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    )


def make_detector(dict_name):
    d = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dict_name))
    p = cv2.aruco.DetectorParameters()
    p.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    return cv2.aruco.ArucoDetector(d, p)


def detect_marker_corners_2d(
    image_paths, dict_name=ARUCO_DICT_NAME, marker_id=MARKER_ID
):
    detector = make_detector(dict_name)
    detections, other_ids = {}, {}
    for path in image_paths:
        img = cv2.imread(path)
        if img is None:
            continue
        corners, ids, _ = detector.detectMarkers(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
        if ids is None:
            continue
        for c, i in zip(corners, ids.flatten()):
            if int(i) == marker_id:
                detections[os.path.basename(path)] = c[0]  # 4x2 pixel coords
            else:
                other_ids[int(i)] = other_ids.get(int(i), 0) + 1
    print(
        f"Marker ID {marker_id} ({dict_name}) detected in {len(detections)} / {len(image_paths)} images"
    )
    if other_ids:
        print(
            f"  (other marker IDs seen: {other_ids} - the printed marker may have a different ID)"
        )
    return detections


def diagnose_dictionaries(image_paths, max_images=12):
    """Called when nothing was found: which ArUco dictionary (if any) matches the photos?"""
    step = max(1, len(image_paths) // max_images)
    sample = image_paths[::step][:max_images]
    print(
        f"\nNo marker found - trying {len(DICT_CANDIDATES)} dictionaries on {len(sample)} sample photos ..."
    )
    found = {}
    for name in DICT_CANDIDATES:
        if not hasattr(cv2.aruco, name):
            continue
        det = make_detector(name)
        hits, ids_seen = 0, set()
        for p in sample:
            img = cv2.imread(p)
            if img is None:
                continue
            _, ids, _ = det.detectMarkers(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
            if ids is not None and len(ids):
                hits += 1
                ids_seen |= set(int(i) for i in ids.flatten())
        if hits:
            found[name] = (hits, sorted(ids_seen))
    if found:
        print("Markers WERE found with these dictionaries (hits, IDs):")
        for k, v in sorted(found.items(), key=lambda kv: -kv[1][0]):
            print(f"   {k}: seen in {v[0]} photos, IDs {v[1]}")
        print(
            "-> Set ARUCO_DICT_NAME / MARKER_ID at the top of this file to match your printed marker."
        )
        print(
            "   (A single hit in an unrelated dictionary can be a false detection; trust the one with most hits.)"
        )
    else:
        print(
            "No ArUco/AprilTag marker found with ANY dictionary. Either the marker is not in these photos,\n"
            "or it is too small / blurred / glary. Open a photo and check that a printed marker is visible\n"
            "and covers at least ~100 pixels across. Otherwise re-shoot with the marker in view."
        )


def triangulate_marker_corners(sparse_path, detections):
    import pycolmap

    reconstruction = pycolmap.Reconstruction(sparse_path)
    name_to_image = {img.name: img for img in reconstruction.images.values()}
    cameras = reconstruction.cameras
    corner_rays = [[] for _ in range(4)]
    used = 0
    for img_name, corners_2d in detections.items():
        if img_name not in name_to_image:
            continue
        image = name_to_image[img_name]
        camera = cameras[image.camera_id]
        cfw = image.cam_from_world
        world_from_cam = (cfw() if callable(cfw) else cfw).inverse()
        cam_center = np.asarray(world_from_cam.translation)
        used += 1
        for i, px in enumerate(corners_2d):
            r = camera.cam_from_img(px)
            ray_world = world_from_cam.rotation * np.array([r[0], r[1], 1.0])
            corner_rays[i].append((cam_center, ray_world / np.linalg.norm(ray_world)))
    print(f"Marker images matched to COLMAP poses: {used}")

    def triangulate_rays(rays):
        A, b = np.zeros((3, 3)), np.zeros(3)
        for o, d in rays:
            d = np.asarray(d).reshape(3, 1)
            P = np.eye(3) - d @ d.T
            A += P
            b += P @ np.asarray(o)
        return np.linalg.solve(A, b)

    out = []
    for i, rays in enumerate(corner_rays):
        if len(rays) < MIN_VIEWS_REQUIRED:
            raise RuntimeError(
                f"Corner {i} only seen in {len(rays)} registered image(s); need {MIN_VIEWS_REQUIRED}. "
                "Capture more overlapping photos with the marker in view."
            )
        out.append(triangulate_rays(rays))
    return np.array(out)


def compute_scale_factor(c3d):
    sides = [np.linalg.norm(c3d[i] - c3d[(i + 1) % 4]) for i in range(4)]
    mean = float(np.mean(sides))
    return (
        MARKER_SIZE_CM / mean,
        mean,
        (float(np.std(sides)) / mean if mean > 0 else 1.0),
    )


def main():
    if os.path.isdir(UNDISTORTED_IMGS) and list_images(UNDISTORTED_IMGS):
        image_dir = UNDISTORTED_IMGS
    else:
        image_dir = ORIGINAL_IMAGES
        print(
            "WARNING: dense/images not found - using the original photos (run 01_reconstruct.py first)."
        )
    paths = list_images(image_dir)
    print(f"Found {len(paths)} images in {image_dir}")

    detections = detect_marker_corners_2d(paths)
    if len(detections) < MIN_VIEWS_REQUIRED:
        diagnose_dictionaries(list_images(ORIGINAL_IMAGES))
        raise SystemExit(
            f"\nMarker ID {MARKER_ID} found in only {len(detections)} image(s); need {MIN_VIEWS_REQUIRED}."
        )

    c3d = triangulate_marker_corners(DENSE_SPARSE_PATH, detections)
    scale, units, disagreement = compute_scale_factor(c3d)
    print(f"\nTriangulated marker side (avg of 4 edges): {units:.5f} units")
    print(f"Side-length disagreement: {disagreement*100:.2f}%   (good: < 2%)")
    print(f"Scale factor: {scale:.4f} cm/unit")
    if disagreement > 0.05:
        print(
            "\nWARNING: sides disagree by >5% - marker not flat, poorly detected, or too small. Do not trust this scale."
        )
    with open(OUTPUT_JSON, "w") as f:
        json.dump(
            {
                "scale_factor_cm_per_unit": scale,
                "measured_marker_units": units,
                "disagreement_pct": disagreement,
                "marker_size_cm": MARKER_SIZE_CM,
                "marker_dictionary": ARUCO_DICT_NAME,
                "marker_id": MARKER_ID,
                "num_images_used": len(detections),
            },
            f,
            indent=2,
        )
    print(f"\nSaved scale factor to {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
