"""
Step 3 (v2): robust pothole measurement from the COLMAP dense cloud
====================================================================
Why a rewrite of 03_measure_pothole.py?  Tests on synthetic potholes of KNOWN size
showed v1 (a) failed ~50% of runs because the RANSAC plane normal has an arbitrary
sign, so "below the road" was sometimes "above" it, and (b) inflated length/width
once reconstruction noise exceeded ~0.1 cm, because noise points were counted as
pothole points.

v2 approach
  1. Fit the road plane (RANSAC + least squares) and ORIENT the normal using the
     camera positions (cameras are above the road).
  2. Build a HEIGHT MAP: project points on the plane, take the MEDIAN depth in every
     2 cm cell. Medians average the noise away, so thresholds can be small.
  3. Pothole = connected group of cells deeper than a noise-aware threshold.
  4. Re-fit the road plane locally (road cells around the pothole only) and repeat,
     which removes bias from road curvature.
  5. Report length/width (min-area rectangle), true AREA (cells x cell area),
     mean and max depth, plus uncertainty ranges and the detection limit.
All internal work is in cm.

Usage:  python 03_measure_pothole_v2.py [--all-clusters] [--mesh]
        --mesh : measure on dense/meshed.ply instead of dense/fused.ply (compare both!)
"""

import os, sys, json
import numpy as np, pandas as pd, cv2, open3d as o3d

PROJECT_DIR = os.environ.get("POTHOLE_PROJECT_DIR", r"D:\MinProject")
FUSED_PLY = os.path.join(PROJECT_DIR, "dense", "fused.ply")
SCALE_JSON = os.path.join(PROJECT_DIR, "scale_factor.json")
SPARSE_DIR = os.path.join(PROJECT_DIR, "dense", "sparse")  # undistorted model
MESHED_PLY = os.path.join(
    PROJECT_DIR, "dense", "meshed.ply"
)  # Poisson mesh (used with --mesh)

GLOBAL_TOL_CM = 0.5  # RANSAC road tolerance
CELL_CM = 2.0  # height-map cell size
MIN_CELL_PTS = 4  # cells with fewer points are treated as "no data"
MIN_DEPTH_CM = 0.4  # never call anything shallower than this a pothole
SIGMA_K = 4.0  # threshold = max(MIN_DEPTH, SIGMA_K x noise of a cell median)
MIN_CELLS = 15  # smallest pothole kept (15 cells = 60 cm2)
MARGIN_CELLS = 8  # road context kept around a pothole for the local plane fit
N_STRIPS = 6


def ls_plane(p):
    c = p.mean(0)
    _, _, vt = np.linalg.svd(p - c, full_matrices=False)
    n = vt[2]
    return n, -float(n @ c)


def orient(n, d, cams):
    return (-n, -d) if np.mean(cams @ n + d) < 0 else (n, d)


def robust_sigma(x):
    return 1.4826 * np.median(np.abs(x - np.median(x)))


def basis(n):
    arb = np.array([1.0, 0, 0]) if abs(n[0]) < 0.9 else np.array([0, 1.0, 0])
    u = np.cross(n, arb)
    u /= np.linalg.norm(u)
    return u, np.cross(n, u)


def depth_grid(pts, n, d, origin, u, v):
    depth = -(pts @ n + d)
    rel = pts - origin
    ij = np.floor(np.stack([rel @ u, rel @ v], 1) / CELL_CM).astype(int)
    mn = ij.min(0)
    ij = ij - mn
    H, W = ij.max(0) + 1
    if H * W > 6e6:
        raise RuntimeError("Point cloud extent too large - crop it first.")
    g = (
        pd.DataFrame({"k": ij[:, 0] * W + ij[:, 1], "d": depth})
        .groupby("k")["d"]
        .agg(["median", "size"])
    )
    grid = np.full(H * W, np.nan)
    cnt = np.zeros(H * W)
    grid[g.index.values] = g["median"].values
    cnt[g.index.values] = g["size"].values
    grid[cnt < MIN_CELL_PTS] = np.nan
    return grid.reshape(H, W), cnt.reshape(H, W), mn, ij


def threshold(cnt, sigma):
    typ = (
        np.median(cnt[cnt >= MIN_CELL_PTS])
        if np.any(cnt >= MIN_CELL_PTS)
        else MIN_CELL_PTS
    )
    return max(MIN_DEPTH_CM, SIGMA_K * sigma / np.sqrt(max(typ, 1.0)))


def components(grid, thr):
    mask = (np.nan_to_num(grid, nan=0.0) > thr).astype(np.uint8)
    nl, lab, stats, cent = cv2.connectedComponentsWithStats(mask, connectivity=8)
    keep = [k for k in range(1, nl) if stats[k, cv2.CC_STAT_AREA] >= MIN_CELLS]
    keep.sort(key=lambda k: -stats[k, cv2.CC_STAT_AREA])
    return lab, keep


def metrics(grid, comp):
    ii, jj = np.nonzero(comp)
    cx = (jj + 0.5) * CELL_CM
    cy = (ii + 0.5) * CELL_CM
    hull = cv2.convexHull(np.stack([cx, cy], 1).astype(np.float32))
    (rx, ry), (w, hh), ang = cv2.minAreaRect(hull)
    w, hh = (
        w + CELL_CM / 2,
        hh + CELL_CM / 2,
    )  # centres of edge cells sit ~half a cell inside the edge
    width, length = sorted([w, hh])
    th = np.deg2rad(ang)
    rot = np.array([[np.cos(th), np.sin(th)], [-np.sin(th), np.cos(th)]])
    loc = (np.stack([cx, cy], 1) - np.array([rx, ry])) @ rot.T
    major = loc[:, 0] if w >= hh else loc[:, 1]
    smooth = cv2.blur(np.nan_to_num(grid, nan=0.0).astype(np.float32), (3, 3))
    depths = smooth[ii, jj]
    edges = np.linspace(major.min(), major.max() + 1e-6, N_STRIPS + 1)
    strip = [
        depths[(major >= edges[s]) & (major < edges[s + 1])].max()
        for s in range(N_STRIPS)
        if np.any((major >= edges[s]) & (major < edges[s + 1]))
    ]
    return dict(
        length=length,
        width=width,
        area=float(comp.sum() * CELL_CM**2),
        depth_mean=float(np.mean(strip)),
        depth_max=float(depths.max()),
        depth_min=float(np.min(strip)),
        strips=[float(s) for s in strip],
        ncells=int(comp.sum()),
    )


def measure_all(pts, cams, scale_unc_pct=0.02, all_clusters=False):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    (a, b, c, d), inl = pcd.segment_plane(GLOBAL_TOL_CM, 3, 2000)
    n, d = orient(*ls_plane(pts[inl]), cams)
    sd = pts @ n + d
    sigma = robust_sigma(sd[np.abs(sd) < 3.0])
    keep_pts = np.abs(sd) < 40.0
    P = pts[keep_pts]
    origin = pts[inl].mean(0)
    u, v = basis(n)
    grid, cnt, mn, ij = depth_grid(P, n, d, origin, u, v)
    thr = threshold(cnt, sigma)
    lab, comps = components(grid, thr)
    if not comps:
        raise RuntimeError(
            f"No pothole deeper than {thr:.2f} cm found (noise sigma {sigma:.2f} cm)."
        )
    out = []
    for idx, k in enumerate(comps if all_clusters else comps[:1]):
        try:
            ci, cj = np.nonzero(lab == k)
            i0, i1 = max(ci.min() - MARGIN_CELLS, 0), ci.max() + MARGIN_CELLS
            j0, j1 = max(cj.min() - MARGIN_CELLS, 0), cj.max() + MARGIN_CELLS
            in_crop = (
                (ij[:, 0] >= i0)
                & (ij[:, 0] <= i1)
                & (ij[:, 1] >= j0)
                & (ij[:, 1] <= j1)
            )
            crop = P[in_crop]
            # road points = crop points NOT near the pothole (dilated by 2 cells)
            dil = cv2.dilate(
                (lab == k).astype(np.uint8), np.ones((5, 5), np.uint8)
            ).astype(bool)
            near = dil[ij[in_crop, 0], ij[in_crop, 1]]
            road = crop[~near]
            road = road[np.abs(road @ n + d) < max(3 * sigma, GLOBAL_TOL_CM) + 1.0]
            if len(road) < 50:
                raise RuntimeError("Too few road points around pothole.")
            n2, d2 = orient(*ls_plane(road), cams)
            for _ in range(2):
                r2 = road[np.abs(road @ n2 + d2) < max(3 * sigma, GLOBAL_TOL_CM)]
                if len(r2) >= 50:
                    n2, d2 = orient(*ls_plane(r2), cams)
            o2 = crop.mean(0)
            u2, v2 = basis(n2)
            g2, c2, mn2, _ = depth_grid(crop, n2, d2, o2, u2, v2)
            thr2 = threshold(c2, sigma)
            lab2, comps2 = components(g2, thr2)
            if not comps2:
                raise RuntimeError("Pothole disappears after local plane fit.")
            cen = pcd_cell = None
            # pick the component nearest the old pothole centre
            c3 = P[ij[:, 0] == -1]  # placeholder (unused)
            cen_pt = P[in_crop][(~near)[:] == False]
            cen_pt = cen_pt.mean(0) if len(cen_pt) else crop.mean(0)
            rel = cen_pt - o2
            ci2 = int(np.floor((rel @ u2) / CELL_CM)) - mn2[0]
            cj2 = int(np.floor((rel @ v2) / CELL_CM)) - mn2[1]
            k2 = comps2[0]
            if (
                0 <= ci2 < lab2.shape[0]
                and 0 <= cj2 < lab2.shape[1]
                and lab2[ci2, cj2] in comps2
            ):
                k2 = lab2[ci2, cj2]
            m = metrics(g2, lab2 == k2)
        except RuntimeError as e:
            print(f"Skipping pothole {idx}: {e}")
            continue
        tol = max(0.03, scale_unc_pct)
        rng = lambda val, floor: (
            val - max(val * tol, floor),
            val + max(val * tol, floor),
        )
        warn = []
        if m["depth_max"] < 2 * thr2:
            warn.append("depth is close to the detection limit; treat as unreliable")
        out.append(
            dict(
                cluster_label=idx if all_clusters else "",
                num_cells=m["ncells"],
                length_cm=m["length"],
                length_range_cm=rng(m["length"], CELL_CM),
                width_cm=m["width"],
                width_range_cm=rng(m["width"], CELL_CM),
                area_cm2=m["area"],
                depth_cm=m["depth_mean"],
                depth_range_cm=rng(m["depth_mean"], max(0.3, thr2 / 2)),
                depth_max_cm=m["depth_max"],
                depth_max_range_cm=rng(m["depth_max"], max(0.3, thr2 / 2)),
                depth_min_cm=m["depth_min"],
                strip_depths_cm=m["strips"],
                noise_sigma_cm=float(sigma),
                detection_threshold_cm=float(thr2),
                warnings=warn,
            )
        )
    return out


def camera_centers(sparse_dir):
    import pycolmap

    rec = pycolmap.Reconstruction(sparse_dir)
    cs = []
    for im in rec.images.values():
        cfw = im.cam_from_world
        cfw = cfw() if callable(cfw) else cfw
        cs.append(np.asarray(cfw.inverse().translation))
    return np.array(cs)


def main():
    sc = json.load(open(SCALE_JSON))
    s = sc["scale_factor_cm_per_unit"]
    unc = max(sc.get("disagreement_pct", 0.02), 0.02)
    if "--mesh" in sys.argv:  # measure on the Poisson mesh vertices instead
        pts = np.asarray(o3d.io.read_triangle_mesh(MESHED_PLY).vertices) * s
        print(f"Using mesh vertices from {MESHED_PLY}: {len(pts)} points")
    else:
        pcd = o3d.io.read_point_cloud(FUSED_PLY)
        pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
        pts = np.asarray(pcd.points) * s  # -> cm
    cams = camera_centers(SPARSE_DIR) * s
    res = measure_all(pts, cams, unc, "--all-clusters" in sys.argv)
    for r in res:
        print(
            f"\nPOTHOLE {r['cluster_label']}: {r['length_cm']:.1f} x {r['width_cm']:.1f} cm, "
            f"area {r['area_cm2']:.0f} cm2, depth mean {r['depth_cm']:.1f} / max {r['depth_max_cm']:.1f} cm "
            f"(noise sigma {r['noise_sigma_cm']:.2f} cm, detection threshold {r['detection_threshold_cm']:.2f} cm)"
        )
        for w in r["warnings"]:
            print("  WARNING:", w)
    json.dump(
        res, open(os.path.join(PROJECT_DIR, "pothole_measurements.json"), "w"), indent=2
    )
    print(f"\nSaved {len(res)} measurement(s).")


if __name__ == "__main__":
    main()
