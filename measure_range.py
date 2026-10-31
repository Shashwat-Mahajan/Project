import open3d as o3d
import numpy as np
import cv2

pcd = o3d.io.read_point_cloud("dense/fused.ply")

# =================================================================
# STEP 1: Denoise
# =================================================================
pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)

# =================================================================
# STEP 2: Scale calibration using MULTIPLE reference objects
# -----------------------------------------------------------------
# Place 2-3 known objects (e.g. A4 sheets, rulers, checkerboard
# squares) around the pothole area (not just one, and not far away —
# closer references = more accurate local scale). Measure each one's
# real size and its size in the point cloud, then average the
# resulting scale factors. This cancels out random measurement error
# instead of trusting a single manual measurement.
# =================================================================
# RECOMMENDED: get this value from scale_calibration_aruco.py (--pick mode)
# using an ArUco marker instead of a paper sheet. Much more accurate since
# marker corners are sharp, flat, and easy to pick precisely.
USE_ARUCO_SCALE = True
ARUCO_SCALE_FACTOR = (
    None  # <-- PASTE the value printed by scale_calibration_aruco.py --pick
)
ARUCO_SCALE_UNCERTAINTY_PCT = 0.02

if USE_ARUCO_SCALE:
    if ARUCO_SCALE_FACTOR is None:
        raise SystemExit(
            "ARUCO_SCALE_FACTOR is not set. Run:\n"
            "  python scale_calibration_aruco.py --pick\n"
            "and paste the printed scale_factor value into this script before running."
        )
    scale_factor = ARUCO_SCALE_FACTOR
    scale_std = ARUCO_SCALE_FACTOR * ARUCO_SCALE_UNCERTAINTY_PCT
    print(f"Scale factor (ArUco): {scale_factor:.3f} cm/unit")
else:
    # Fallback: multiple manual reference objects (less accurate)
    references = [
        (21.0, 0.42),
        (29.7, 0.595),
    ]
    scale_factors = [real / measured for real, measured in references]
    scale_factor = float(np.mean(scale_factors))
    scale_std = float(np.std(scale_factors))
    print(
        f"Scale factor (manual refs): {scale_factor:.3f} cm/unit  (std: {scale_std:.3f})"
    )
    if scale_std / scale_factor > 0.05:
        print("Warning: reference measurements disagree by >5% - re-check them.")

# =================================================================
# STEP 3: Segment road surface using RANSAC — LOCAL to the pothole
# -----------------------------------------------------------------
# Fitting the plane on the WHOLE road picks up global curvature/slope
# and biases the plane away from the pothole's true local surface.
# Crop a region around the pothole first (rough manual bounds), fit
# the plane on that local patch, then apply it to the pothole points.
# =================================================================
all_points = np.asarray(pcd.points)

bbox_min = all_points.min(axis=0)
bbox_max = all_points.max(axis=0)
print("Point cloud bounds:")
print("  min =", bbox_min)
print("  max =", bbox_max)

# -----------------------------------------------------------------
# CRITICAL: you MUST narrow this to just the pothole + surrounding
# road, or the "pothole" ends up being your entire scene (this is
# what caused the 534cm-wide result last run).
#
# EASIEST WAY: run with --pickcrop once to interactively select the
# pothole region in Open3D, which prints the exact CROP_MIN/CROP_MAX
# to paste in below.
# -----------------------------------------------------------------
import sys

if "--pickcrop" in sys.argv:
    print("\nShift+click 2 opposite corners of a box around the pothole")
    print("(just rough corners of the region containing pothole + nearby road).")
    print("Then close the window (press Q).")
    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window(width=1280, height=800)
    vis.add_geometry(pcd)
    vis.get_render_option().point_size = 3.0
    vis.run()
    vis.destroy_window()
    picked = vis.get_picked_points()
    if len(picked) < 2:
        raise SystemExit("Need at least 2 points picked to define a crop box. Re-run.")
    picked_pts = all_points[picked[:2]]
    crop_min_picked = picked_pts.min(axis=0)
    crop_max_picked = picked_pts.max(axis=0)
    print(f"\nCROP_MIN = np.array({list(crop_min_picked)})")
    print(f"CROP_MAX = np.array({list(crop_max_picked)})")
    print(
        "\n--> Paste these two lines into the CROP_MIN/CROP_MAX section below, "
        "then re-run WITHOUT --pickcrop."
    )
    raise SystemExit(0)

# PASTE the CROP_MIN / CROP_MAX printed by --pickcrop here.
# Defaults to full cloud (WRONG for actual measurement — only safe
# for a first sanity-check run to confirm the script executes).
CROP_MIN = bbox_min.copy()
CROP_MAX = bbox_max.copy()

crop_mask = np.all((all_points >= CROP_MIN) & (all_points <= CROP_MAX), axis=1)
n_cropped = int(crop_mask.sum())
print("Points in crop region:", n_cropped)

if np.allclose(CROP_MIN, bbox_min) and np.allclose(CROP_MAX, bbox_max):
    print("\n" + "=" * 70)
    print("WARNING: CROP_MIN/CROP_MAX still equal the FULL point cloud bounds.")
    print("Results below will measure your ENTIRE SCENE, not the pothole,")
    print("and will be meaningless (this is what caused the 534cm-wide result).")
    print("Run:  python measure_range.py --pickcrop")
    print("to select just the pothole region first.")
    print("=" * 70 + "\n")

if n_cropped < 3:
    raise SystemExit(
        "Crop region contains too few points. Check CROP_MIN/CROP_MAX "
        "against the printed point cloud bounds above."
    )

local_cloud = pcd.select_by_index(np.where(crop_mask)[0])

plane_model, inliers = local_cloud.segment_plane(
    distance_threshold=0.01, ransac_n=3, num_iterations=2000
)
a, b, c, d = plane_model

road_cloud = local_cloud.select_by_index(inliers)
pothole_cloud = local_cloud.select_by_index(inliers, invert=True)
pothole_points = np.asarray(pothole_cloud.points)

# =================================================================
# STEP 4: Keep only points clearly BELOW the local road plane
# =================================================================
signed_dist = (
    a * pothole_points[:, 0] + b * pothole_points[:, 1] + c * pothole_points[:, 2] + d
)
below_road = pothole_points[signed_dist < -0.005]

if len(below_road) == 0:
    print(
        "No pothole points found below road plane. Tune distance_threshold or crop bounds."
    )
else:
    # =============================================================
    # STEP 5: Project pothole points onto the road plane (2D),
    # then use a ROTATED minimum-area bounding box instead of
    # axis-aligned max-min. This matches the pothole's actual
    # orientation and shape far better than X/Y extents.
    # =============================================================
    normal = np.array([a, b, c])
    normal = normal / np.linalg.norm(normal)

    # Build an orthonormal basis (u, v) on the plane
    arbitrary = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(arbitrary, normal)) > 0.9:
        arbitrary = np.array([0.0, 1.0, 0.0])
    u = np.cross(normal, arbitrary)
    u = u / np.linalg.norm(u)
    v = np.cross(normal, u)

    # Project each point onto the plane's local 2D (u, v) coordinates
    origin = below_road.mean(axis=0)
    rel = below_road - origin
    coords_2d = np.stack([rel @ u, rel @ v], axis=1).astype(np.float32)

    # Convex hull + rotated min-area rectangle (OpenCV)
    hull = cv2.convexHull(coords_2d)
    (cx, cy), (w_units, h_units), angle = cv2.minAreaRect(hull)

    width_units, length_units = sorted([w_units, h_units])  # shorter, longer

    # =============================================================
    # STEP 6: Cross-sectional depth profiling
    # -------------------------------------------------------------
    # Instead of one single "deepest point" number (sensitive to a
    # single noisy point), slice the pothole into parallel strips
    # along its length and compute the max depth PER STRIP. This is
    # how pavement condition studies actually report pothole depth,
    # and it also gives you a depth PROFILE (useful for volume
    # estimates and much more defensible than one number).
    # =============================================================
    depths_all = -(
        a * below_road[:, 0] + b * below_road[:, 1] + c * below_road[:, 2] + d
    )

    # Use the 2D projected coords (already computed as coords_2d) to
    # slice along the LONGER axis of the rotated bounding box.
    # Re-derive per-point positions along the box's major axis.
    box_angle_rad = np.deg2rad(angle)
    rot = np.array(
        [
            [np.cos(box_angle_rad), np.sin(box_angle_rad)],
            [-np.sin(box_angle_rad), np.cos(box_angle_rad)],
        ]
    )
    local_coords = (coords_2d - np.array([cx, cy])) @ rot.T
    # local_coords[:,0] and [:,1] are now aligned to the box's edges

    major_axis = local_coords[:, 0] if w_units >= h_units else local_coords[:, 1]

    N_STRIPS = 6
    strip_edges = np.linspace(major_axis.min(), major_axis.max(), N_STRIPS + 1)
    strip_max_depths = []
    for i in range(N_STRIPS):
        mask = (major_axis >= strip_edges[i]) & (major_axis < strip_edges[i + 1])
        if mask.sum() > 0:
            strip_max_depths.append(depths_all[mask].max())

    strip_max_depths = np.array(strip_max_depths)
    depth_units_mean = float(strip_max_depths.mean())
    depth_units_max = float(strip_max_depths.max())
    depth_units_min = (
        float(strip_max_depths[strip_max_depths > 0].min())
        if np.any(strip_max_depths > 0)
        else 0.0
    )

    width_cm = width_units * scale_factor
    length_cm = length_units * scale_factor
    depth_cm_mean = depth_units_mean * scale_factor
    depth_cm_max = depth_units_max * scale_factor
    depth_cm_min = depth_units_min * scale_factor
    depth_cm = depth_cm_mean  # use strip-averaged depth as the headline value

    # =============================================================
    # STEP 7: Report as a small range reflecting real uncertainty
    # sources: scale calibration spread + a fixed measurement floor
    # =============================================================
    TOLERANCE_PCT = max(
        0.03, scale_std / scale_factor
    )  # at least 3%, or scale disagreement
    MIN_MARGIN_CM = 0.3

    def as_range(value, pct=TOLERANCE_PCT, min_margin=MIN_MARGIN_CM):
        margin = max(value * pct, min_margin)
        return value - margin, value + margin

    width_low, width_high = as_range(width_cm)
    length_low, length_high = as_range(length_cm)
    depth_low, depth_high = as_range(depth_cm)

    print(f"\n📏 POTHOLE MEASUREMENTS (rotated bbox + cross-sectional depth):")
    print(f"Width :  {width_low:.1f} - {width_high:.1f} cm  (est. {width_cm:.1f} cm)")
    print(
        f"Length:  {length_low:.1f} - {length_high:.1f} cm  (est. {length_cm:.1f} cm)"
    )
    print(
        f"Depth :  {depth_low:.1f} - {depth_high:.1f} cm  (est. {depth_cm:.1f} cm, "
        f"strip range {depth_cm_min:.1f}-{depth_cm_max:.1f} cm across {N_STRIPS} slices)"
    )

    print(f"\nDepth profile across {N_STRIPS} cross-sections (shallow -> deep end):")
    for i, sd in enumerate(strip_max_depths):
        print(f"  Slice {i+1}: {sd * scale_factor:.1f} cm")
