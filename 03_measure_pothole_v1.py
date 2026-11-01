"""
Step 3: Fully autonomous pothole detection + measurement
===========================================================
No manual point-picking. Reads the scale factor from step 2, then:
  1. Denoises the dense point cloud.
  2. Fits a global road plane with RANSAC (thresholds derived from
     scale_factor so they're physically meaningful, not arbitrary).
  3. Finds all points meaningfully below that plane.
  4. Clusters those points with DBSCAN; the largest cluster is
     treated as the pothole.
  5. Re-fits a LOCAL plane using only the region immediately around
     the detected pothole (avoids global road curvature bias).
  6. Projects pothole points onto the local plane, computes a
     rotated minimum-area bounding box for width/length, and slices
     the pothole into cross-sections for a robust depth profile.
  7. Reports results as a range reflecting scale + fit uncertainty.

Run after 02_scale_calibration.py:
    python 03_measure_pothole.py

If your road segment may contain MULTIPLE potholes, see
`--all-clusters` at the bottom to measure every cluster above a
minimum size instead of just the largest.
"""

import os
import sys
import json
import numpy as np
import open3d as o3d
import cv2

# -----------------------------------------------------------------
# CONFIG
# -----------------------------------------------------------------
PROJECT_DIR = r"D:\MinProject"
FUSED_PLY = os.path.join(PROJECT_DIR, "dense", "fused.ply")
SCALE_JSON = os.path.join(PROJECT_DIR, "scale_factor.json")

# Physical thresholds in cm - converted to point-cloud units using scale_factor
GLOBAL_PLANE_TOL_CM = 0.5  # road roughness tolerance for rough global plane
BELOW_ROAD_CUTOFF_CM = 1.0  # min depth below global plane to be a "candidate"
DBSCAN_NEIGHBOR_CM = 5.0  # cluster neighbor distance
DBSCAN_MIN_POINTS = 15
MIN_CLUSTER_POINTS = 20  # ignore tiny noise clusters
CROP_MARGIN_CM = 15.0  # extra road context around detected pothole
LOCAL_PLANE_TOL_CM = 0.3  # tighter tolerance for the local re-fit
LOCAL_BELOW_CUTOFF_CM = 0.2  # min depth below LOCAL plane to count as pothole
N_STRIPS = 6  # cross-sectional depth slices


def load_scale_factor():
    if not os.path.exists(SCALE_JSON):
        raise SystemExit(f"{SCALE_JSON} not found. Run 02_scale_calibration.py first.")
    with open(SCALE_JSON) as f:
        data = json.load(f)
    return data["scale_factor_cm_per_unit"], data


def auto_detect_clusters(pcd, scale_factor):
    """Returns list of (points, label) for below-road clusters, largest first."""
    points = np.asarray(pcd.points)

    plane_model, _ = pcd.segment_plane(
        distance_threshold=GLOBAL_PLANE_TOL_CM / scale_factor,
        ransac_n=3,
        num_iterations=2000,
    )
    a, b, c, d = plane_model
    signed_dist = a * points[:, 0] + b * points[:, 1] + c * points[:, 2] + d
    below_mask = signed_dist < -(BELOW_ROAD_CUTOFF_CM / scale_factor)
    below_points = points[below_mask]

    print(f"Global plane fit: {below_mask.sum()} candidate below-road points")

    if len(below_points) < MIN_CLUSTER_POINTS:
        raise RuntimeError(
            "No significant depression found below the global road plane. "
            "Check BELOW_ROAD_CUTOFF_CM / GLOBAL_PLANE_TOL_CM, or verify the "
            "pothole is actually present in this scan."
        )

    below_pcd = o3d.geometry.PointCloud()
    below_pcd.points = o3d.utility.Vector3dVector(below_points)

    labels = np.array(
        below_pcd.cluster_dbscan(
            eps=DBSCAN_NEIGHBOR_CM / scale_factor,
            min_points=DBSCAN_MIN_POINTS,
        )
    )

    if labels.max() < 0:
        raise RuntimeError(
            "DBSCAN found no clusters. Try increasing DBSCAN_NEIGHBOR_CM or "
            "lowering DBSCAN_MIN_POINTS."
        )

    clusters = []
    for label in range(labels.max() + 1):
        cluster_pts = below_points[labels == label]
        if len(cluster_pts) >= MIN_CLUSTER_POINTS:
            clusters.append(cluster_pts)

    clusters.sort(key=len, reverse=True)
    print(
        f"Found {len(clusters)} candidate pothole cluster(s) "
        f"(sizes: {[len(c) for c in clusters]})"
    )
    return clusters


def measure_single_pothole(pcd, cluster_pts, scale_factor, scale_std, label=""):
    all_points = np.asarray(pcd.points)

    margin_units = CROP_MARGIN_CM / scale_factor
    crop_min = cluster_pts.min(axis=0) - margin_units
    crop_max = cluster_pts.max(axis=0) + margin_units

    crop_mask = np.all((all_points >= crop_min) & (all_points <= crop_max), axis=1)
    local_cloud = pcd.select_by_index(np.where(crop_mask)[0])

    if len(local_cloud.points) < 50:
        raise RuntimeError(
            f"Local crop around cluster{label} has too few points "
            f"({len(local_cloud.points)}). Increase CROP_MARGIN_CM."
        )

    # Re-fit a LOCAL road plane (avoids global curvature bias)
    plane_model, inliers = local_cloud.segment_plane(
        distance_threshold=LOCAL_PLANE_TOL_CM / scale_factor,
        ransac_n=3,
        num_iterations=2000,
    )
    a, b, c, d = plane_model

    pothole_cloud = local_cloud.select_by_index(inliers, invert=True)
    pothole_points = np.asarray(pothole_cloud.points)

    signed_dist = (
        a * pothole_points[:, 0]
        + b * pothole_points[:, 1]
        + c * pothole_points[:, 2]
        + d
    )
    below_road = pothole_points[signed_dist < -(LOCAL_BELOW_CUTOFF_CM / scale_factor)]

    if len(below_road) == 0:
        raise RuntimeError(
            f"No pothole points found below local road plane for cluster{label}. "
            "Tune LOCAL_BELOW_CUTOFF_CM."
        )

    normal = np.array([a, b, c])
    normal = normal / np.linalg.norm(normal)
    arbitrary = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(arbitrary, normal)) > 0.9:
        arbitrary = np.array([0.0, 1.0, 0.0])
    u = np.cross(normal, arbitrary)
    u = u / np.linalg.norm(u)
    v = np.cross(normal, u)

    origin = below_road.mean(axis=0)
    rel = below_road - origin
    coords_2d = np.stack([rel @ u, rel @ v], axis=1).astype(np.float32)

    hull = cv2.convexHull(coords_2d)
    (cx, cy), (w_units, h_units), angle = cv2.minAreaRect(hull)
    width_units, length_units = sorted([w_units, h_units])

    depths_all = -(
        a * below_road[:, 0] + b * below_road[:, 1] + c * below_road[:, 2] + d
    )

    box_angle_rad = np.deg2rad(angle)
    rot = np.array(
        [
            [np.cos(box_angle_rad), np.sin(box_angle_rad)],
            [-np.sin(box_angle_rad), np.cos(box_angle_rad)],
        ]
    )
    local_coords = (coords_2d - np.array([cx, cy])) @ rot.T
    major_axis = local_coords[:, 0] if w_units >= h_units else local_coords[:, 1]

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

    tolerance_pct = max(0.03, scale_std / scale_factor)
    min_margin_cm = 0.3

    def as_range(value):
        margin = max(value * tolerance_pct, min_margin_cm)
        return value - margin, value + margin

    width_low, width_high = as_range(width_cm)
    length_low, length_high = as_range(length_cm)
    depth_low, depth_high = as_range(depth_cm_mean)

    result = {
        "cluster_label": label,
        "num_points": len(below_road),
        "width_cm": width_cm,
        "width_range_cm": (width_low, width_high),
        "length_cm": length_cm,
        "length_range_cm": (length_low, length_high),
        "depth_cm": depth_cm_mean,
        "depth_range_cm": (depth_low, depth_high),
        "depth_min_cm": depth_cm_min,
        "depth_max_cm": depth_cm_max,
        "strip_depths_cm": (strip_max_depths * scale_factor).tolist(),
    }
    return result


def print_result(result):
    label = result["cluster_label"]
    print(f"\n📏 POTHOLE MEASUREMENT{f' [cluster {label}]' if label != '' else ''}:")
    w_lo, w_hi = result["width_range_cm"]
    l_lo, l_hi = result["length_range_cm"]
    d_lo, d_hi = result["depth_range_cm"]
    print(f"Width : {w_lo:.1f} - {w_hi:.1f} cm  (est. {result['width_cm']:.1f} cm)")
    print(f"Length: {l_lo:.1f} - {l_hi:.1f} cm  (est. {result['length_cm']:.1f} cm)")
    print(
        f"Depth : {d_lo:.1f} - {d_hi:.1f} cm  (est. {result['depth_cm']:.1f} cm, "
        f"strip range {result['depth_min_cm']:.1f}-{result['depth_max_cm']:.1f} cm)"
    )
    print(f"Depth profile across {N_STRIPS} cross-sections (shallow -> deep end):")
    for i, sd in enumerate(result["strip_depths_cm"]):
        print(f"  Slice {i+1}: {sd:.1f} cm")


def main():
    scale_factor, scale_data = load_scale_factor()
    scale_std = scale_factor * max(scale_data.get("disagreement_pct", 0.02), 0.02)
    print(
        f"Using scale factor: {scale_factor:.4f} cm/unit "
        f"(from {scale_data.get('num_images_used', '?')} marker images)"
    )

    pcd = o3d.io.read_point_cloud(FUSED_PLY)
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    print(f"Loaded and denoised point cloud: {len(pcd.points)} points")

    clusters = auto_detect_clusters(pcd, scale_factor)

    measure_all = "--all-clusters" in sys.argv
    targets = clusters if measure_all else clusters[:1]

    results = []
    for i, cluster_pts in enumerate(targets):
        label = i if measure_all else ""
        try:
            result = measure_single_pothole(
                pcd, cluster_pts, scale_factor, scale_std, label
            )
            print_result(result)
            results.append(result)
        except RuntimeError as e:
            print(f"\n⚠️  Skipping cluster {i}: {e}")

    out_path = os.path.join(PROJECT_DIR, "pothole_measurements.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n✅ Saved {len(results)} measurement(s) to {out_path}")


if __name__ == "__main__":
    main()
