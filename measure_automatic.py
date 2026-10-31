"""
Fully automated pothole measurement - NO manual picking required.
====================================================================
Instead of you manually selecting the pothole region or marker
corners, this script finds the pothole automatically using
clustering, and uses a simple single-reference scale (same as your
original approach) which only needs you to LOOK at the point cloud
(rotate/zoom is fine) and estimate one measurement -- no clicking.
"""

import open3d as o3d
import numpy as np

INPUT_PATH = r"D:\MinProject\dense\fused.ply"

pcd = o3d.io.read_point_cloud(INPUT_PATH)
pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
points = np.asarray(pcd.points)
print(f"Loaded {len(points)} points (after denoising)")

# =====================================================================
# STEP 1: Scale factor -- single reference measurement (simple, no
# clicking needed in 3D; you can eyeball this from your photos instead,
# since you know the real width of something in the scene, e.g. a tile,
# a curb, a car tire (~20-23cm section width), a standard paver, etc.
# If you have NO physical reference object, use a KNOWN standard
# instead, e.g.:
#   - Indian brick paver: commonly 20 x 10 cm
#   - Standard road lane width: ~350 cm (only useful for large scenes)
#   - A4 paper: 21 x 29.7 cm
# =====================================================================
REFERENCE_REAL_CM = 21.0  # <-- real size of a visible object
REFERENCE_MEASURED_UNITS = 0.42  # <-- rough size of that SAME object
#     in the point cloud units (view
#     the cloud with o3d.visualization
#     .draw_geometries([pcd]) and just
#     eyeball/estimate this from the
#     overall scene scale -- doesn't
#     need pixel-perfect clicking)
scale_factor = REFERENCE_REAL_CM / REFERENCE_MEASURED_UNITS
print(f"Scale factor: {scale_factor:.2f} cm/unit")

# =====================================================================
# STEP 2: Automatically segment the road plane (RANSAC)
# =====================================================================
plane_model, inliers = pcd.segment_plane(
    distance_threshold=0.01, ransac_n=3, num_iterations=2000
)
a, b, c, d = plane_model

non_road_cloud = pcd.select_by_index(inliers, invert=True)
non_road_points = np.asarray(non_road_cloud.points)

signed_dist = (
    a * non_road_points[:, 0]
    + b * non_road_points[:, 1]
    + c * non_road_points[:, 2]
    + d
)
below_road = non_road_points[signed_dist < -0.005]
print(f"Points below road plane (whole scene): {len(below_road)}")

if len(below_road) < 10:
    raise SystemExit(
        "Too few below-road points found. Try lowering the "
        "distance_threshold above (e.g. 0.005) and re-run."
    )

# =====================================================================
# STEP 2b: Restrict candidates to points spatially NEAR the road
# -----------------------------------------------------------------
# Your scene likely contains other structures below the fitted plane
# (walls, curbs, background objects) that are NOT the pothole. Rather
# than clustering the whole scene, project everything into the road
# plane's own 2D coordinates and keep only points that fall within
# (or just outside) the road's own footprint -- the pothole must be
# spatially adjacent to the actual road surface, unlike unrelated
# background geometry.
# =====================================================================
road_points = np.asarray(pcd.select_by_index(inliers).points)

normal0 = np.array([a, b, c])
normal0 = normal0 / np.linalg.norm(normal0)
arb0 = np.array([1.0, 0.0, 0.0])
if abs(np.dot(arb0, normal0)) > 0.9:
    arb0 = np.array([0.0, 1.0, 0.0])
u0 = np.cross(normal0, arb0)
u0 /= np.linalg.norm(u0)
v0 = np.cross(normal0, u0)

origin0 = road_points.mean(axis=0)
road_uv = np.stack([(road_points - origin0) @ u0, (road_points - origin0) @ v0], axis=1)
below_uv = np.stack([(below_road - origin0) @ u0, (below_road - origin0) @ v0], axis=1)

# Road footprint bounding box (with a small margin) in plane coordinates
MARGIN_UNITS = 0.05  # small margin beyond the road's own footprint; tune if needed
road_min = road_uv.min(axis=0) - MARGIN_UNITS
road_max = road_uv.max(axis=0) + MARGIN_UNITS

near_road_mask = np.all((below_uv >= road_min) & (below_uv <= road_max), axis=1)
below_road_near = below_road[near_road_mask]
print(f"Points below road AND within road footprint: {len(below_road_near)}")

if len(below_road_near) < 10:
    raise SystemExit(
        "Too few points near the road footprint. The road plane "
        "fit may be wrong, or MARGIN_UNITS may need adjusting."
    )

below_road = below_road_near

# =====================================================================
# STEP 3: AUTOMATICALLY isolate the pothole using clustering
# -----------------------------------------------------------------
# The below-road points may include noise scattered across the whole
# scene, not just the pothole. DBSCAN groups nearby points into
# clusters; we keep only the LARGEST cluster, which -- assuming your
# scene has one dominant pothole -- will be the pothole itself. This
# replaces manual cropping entirely.
# =====================================================================
below_cloud = o3d.geometry.PointCloud()
below_cloud.points = o3d.utility.Vector3dVector(below_road)

# -----------------------------------------------------------------
# Auto-estimate eps from actual point spacing instead of guessing a
# fixed number. We compute each point's distance to its nearest
# neighbor, then use a multiple of the median as eps -- this adapts
# to YOUR point cloud's density instead of assuming a fixed scale.
# -----------------------------------------------------------------
kdtree = o3d.geometry.KDTreeFlann(below_cloud)
nn_dists = []
sample_idx = np.random.choice(
    len(below_road), size=min(2000, len(below_road)), replace=False
)
for i in sample_idx:
    [_, idx, dist2] = kdtree.search_knn_vector_3d(
        below_road[i], 2
    )  # self + nearest neighbor
    if len(dist2) > 1:
        nn_dists.append(np.sqrt(dist2[1]))
nn_dists = np.array(nn_dists)
median_nn_dist = float(np.median(nn_dists))
EPS = median_nn_dist * 8  # multiple of typical point spacing; tune if needed
MIN_POINTS = 15

print(f"Median nearest-neighbor distance: {median_nn_dist:.5f} units")
print(f"Auto-selected DBSCAN eps: {EPS:.5f} units (min_points={MIN_POINTS})")

labels = np.array(
    below_cloud.cluster_dbscan(eps=EPS, min_points=MIN_POINTS, print_progress=False)
)

if labels.max() < 0:
    raise SystemExit(
        "DBSCAN still found no clusters even with auto eps. "
        "Try manually setting EPS higher (e.g. multiply by 15-20 "
        "instead of 8) and re-run."
    )

unique_labels, counts = np.unique(labels[labels >= 0], return_counts=True)
print(f"Found {len(unique_labels)} clusters below the road plane.")
sorted_idx = np.argsort(-counts)
print("Top 5 cluster sizes:", counts[sorted_idx[:5]].tolist())

largest_label = unique_labels[sorted_idx[0]]
pothole_points = below_road[labels == largest_label]
print(f"Largest cluster (assumed = pothole): {len(pothole_points)} points")

if len(pothole_points) < 30:
    print("\n⚠ WARNING: largest cluster is still very small (<30 points).")
    print("  This measurement will be unreliable. Try increasing the")
    print("  EPS multiplier above (e.g. from 8 to 15) and re-run.")

# =====================================================================
# STEP 4: Compute dimensions using rotated bounding box + depth profile
# =====================================================================
# =====================================================================
# STEP 4: Compute dimensions using rotated bounding box + depth profile
# (pure NumPy implementation -- no cv2/opencv dependency needed)
# =====================================================================


def convex_hull_2d(pts):
    """Andrew's monotone chain convex hull algorithm."""
    pts = np.unique(pts, axis=0)
    if len(pts) < 3:
        return pts
    pts = pts[np.lexsort((pts[:, 1], pts[:, 0]))]

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper = []
    for p in pts[::-1]:
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return np.array(lower[:-1] + upper[:-1])


def min_area_rect(hull):
    """Rotating calipers: minimum-area bounding rectangle of a convex hull.
    Returns (center_x, center_y, width, height, angle_degrees)."""
    n = len(hull)
    if n < 3:
        mins = hull.min(axis=0)
        maxs = hull.max(axis=0)
        c = (mins + maxs) / 2
        w, h = maxs - mins
        return c[0], c[1], w, h, 0.0

    best_area = np.inf
    best = None
    for i in range(n):
        p1, p2 = hull[i], hull[(i + 1) % n]
        edge = p2 - p1
        edge_len = np.linalg.norm(edge)
        if edge_len < 1e-9:
            continue
        edge_dir = edge / edge_len
        perp_dir = np.array([-edge_dir[1], edge_dir[0]])

        proj_edge = hull @ edge_dir
        proj_perp = hull @ perp_dir

        w = proj_edge.max() - proj_edge.min()
        h = proj_perp.max() - proj_perp.min()
        area = w * h

        if area < best_area:
            best_area = area
            cx = (proj_edge.max() + proj_edge.min()) / 2
            cy = (proj_perp.max() + proj_perp.min()) / 2
            center = cx * edge_dir + cy * perp_dir
            angle = np.degrees(np.arctan2(edge_dir[1], edge_dir[0]))
            best = (center[0], center[1], w, h, angle)

    return best


normal = np.array([a, b, c])
normal = normal / np.linalg.norm(normal)
arbitrary = np.array([1.0, 0.0, 0.0])
if abs(np.dot(arbitrary, normal)) > 0.9:
    arbitrary = np.array([0.0, 1.0, 0.0])
u = np.cross(normal, arbitrary)
u = u / np.linalg.norm(u)
v = np.cross(normal, u)

origin = pothole_points.mean(axis=0)
rel = pothole_points - origin
coords_2d = np.stack([rel @ u, rel @ v], axis=1)

hull = convex_hull_2d(coords_2d)
cx, cy, w_units, h_units, angle = min_area_rect(hull)
width_units, length_units = sorted([w_units, h_units])

depths_all = -(
    a * pothole_points[:, 0] + b * pothole_points[:, 1] + c * pothole_points[:, 2] + d
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

N_STRIPS = 6
strip_edges = np.linspace(major_axis.min(), major_axis.max(), N_STRIPS + 1)
strip_max_depths = []
for i in range(N_STRIPS):
    mask = (major_axis >= strip_edges[i]) & (major_axis < strip_edges[i + 1])
    if mask.sum() > 0:
        strip_max_depths.append(depths_all[mask].max())
strip_max_depths = np.array(strip_max_depths)

width_cm = width_units * scale_factor
length_cm = length_units * scale_factor
depth_cm_mean = float(strip_max_depths.mean()) * scale_factor
depth_cm_min = float(strip_max_depths.min()) * scale_factor
depth_cm_max = float(strip_max_depths.max()) * scale_factor

TOLERANCE_PCT = 0.05
MIN_MARGIN_CM = 0.3


def as_range(value, pct=TOLERANCE_PCT, mn=MIN_MARGIN_CM):
    m = max(value * pct, mn)
    return value - m, value + m


w_lo, w_hi = as_range(width_cm)
l_lo, l_hi = as_range(length_cm)
d_lo, d_hi = as_range(depth_cm_mean)

print(f"\n{'='*60}")
print(f"POTHOLE MEASUREMENTS (fully automatic)")
print(f"{'='*60}")
print(f"Width :  {w_lo:.1f} - {w_hi:.1f} cm  (est. {width_cm:.1f} cm)")
print(f"Length:  {l_lo:.1f} - {l_hi:.1f} cm  (est. {length_cm:.1f} cm)")
print(
    f"Depth :  {d_lo:.1f} - {d_hi:.1f} cm  (est. {depth_cm_mean:.1f} cm, "
    f"range {depth_cm_min:.1f}-{depth_cm_max:.1f} cm across {N_STRIPS} slices)"
)

# Sanity check -- flag obviously wrong results
if width_cm > 300 or length_cm > 300:
    print("\n⚠ WARNING: dimensions look too large for a pothole.")
    print("  The largest cluster may not actually be the pothole.")
    print("  Try adjusting DBSCAN eps/min_points, or check scale_factor.")
