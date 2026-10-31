"""
Diagnostic: why isn't the ArUco marker being detected?
==========================================================
Tries EVERY common ArUco dictionary (not just DICT_4X4_50) against
every image, at multiple detector parameter settings, and saves
annotated copies so you can visually confirm whether a marker is
even present in your photos.

Run:
    python diagnose_marker.py
Check the "marker_debug" folder it creates next to your images.
"""

import os
import glob
import cv2

PROJECT_DIR = r"D:\MinProject"
IMAGE_DIR = os.path.join(PROJECT_DIR, "images")
DEBUG_DIR = os.path.join(PROJECT_DIR, "marker_debug")

# Try all standard dictionaries, not just 4X4_50
DICTS_TO_TRY = {
    "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
    "DICT_4X4_100": cv2.aruco.DICT_4X4_100,
    "DICT_5X5_50": cv2.aruco.DICT_5X5_50,
    "DICT_5X5_100": cv2.aruco.DICT_5X5_100,
    "DICT_6X6_50": cv2.aruco.DICT_6X6_50,
    "DICT_6X6_100": cv2.aruco.DICT_6X6_100,
    "DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL,
    "DICT_APRILTAG_36h11": cv2.aruco.DICT_APRILTAG_36h11,
}


def main():
    os.makedirs(DEBUG_DIR, exist_ok=True)

    image_paths = sorted(
        glob.glob(os.path.join(IMAGE_DIR, "*.jpg"))
        + glob.glob(os.path.join(IMAGE_DIR, "*.JPG"))
        + glob.glob(os.path.join(IMAGE_DIR, "*.jpeg"))
        + glob.glob(os.path.join(IMAGE_DIR, "*.JPEG"))
        + glob.glob(os.path.join(IMAGE_DIR, "*.png"))
        + glob.glob(os.path.join(IMAGE_DIR, "*.PNG"))
    )
    print(f"Found {len(image_paths)} images\n")

    any_hit = False
    for path in image_paths:
        img = cv2.imread(path)
        if img is None:
            print(f"  [SKIP] Could not read {path}")
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        hits_for_image = []
        for dict_name, dict_id in DICTS_TO_TRY.items():
            aruco_dict = cv2.aruco.getPredefinedDictionary(dict_id)
            params = cv2.aruco.DetectorParameters()
            # Loosen thresholds - defaults can miss small/angled/blurry markers
            params.adaptiveThreshWinSizeMin = 3
            params.adaptiveThreshWinSizeMax = 53
            params.adaptiveThreshWinSizeStep = 4
            params.minMarkerPerimeterRate = 0.01  # allow much smaller markers
            detector = cv2.aruco.ArucoDetector(aruco_dict, params)

            corners, ids, _ = detector.detectMarkers(gray)
            if ids is not None and len(ids) > 0:
                hits_for_image.append(dict_name)
                any_hit = True
                annotated = cv2.aruco.drawDetectedMarkers(img.copy(), corners, ids)
                out_name = (
                    f"{os.path.splitext(os.path.basename(path))[0]}_{dict_name}.jpg"
                )
                cv2.imwrite(os.path.join(DEBUG_DIR, out_name), annotated)

        name = os.path.basename(path)
        if hits_for_image:
            print(f"  [HIT]  {name}: detected with {hits_for_image}")
        else:
            print(f"  [MISS] {name}: no marker found in any dictionary")

    print(f"\nAnnotated hits (if any) saved to: {DEBUG_DIR}")
    if not any_hit:
        print(
            "\nNo marker detected in ANY image with ANY dictionary.\n"
            "This almost certainly means no ArUco marker is physically present\n"
            "in these photos. You have two options:\n"
            "  1) Recapture this scene with a printed ArUco marker placed next\n"
            "     to the pothole, then rerun the full pipeline.\n"
            "  2) Use the manual/fallback scale method (measure a known object\n"
            "     already visible in your existing photos, e.g. a ruler, coin,\n"
            "     phone, or the pothole survey stick) instead of ArUco.\n"
        )


if __name__ == "__main__":
    main()
