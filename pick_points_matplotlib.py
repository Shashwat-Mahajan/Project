"""
Matplotlib-based point picker
================================
A reliable alternative when CloudCompare and Open3D's interactive
pickers both fail. This shows the point cloud from a top-down view
(and a side view) using matplotlib, and prints the coordinates of
points near where you click.

Usage:
  python pick_points_matplotlib.py

Controls:
  - Left-click near a point in the TOP-DOWN view (left plot) to find
    its coordinates.
  - The console prints the nearest actual point's X, Y, Z.
  - Close the window when done.
"""

import numpy as np
import matplotlib.pyplot as plt

INPUT_PATH = r"D:\MinProject\dense\fused.xyz"

points = np.loadtxt(INPUT_PATH)
print(f"Loaded {len(points)} points")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))

# Top-down view (X vs Y)
ax1.scatter(points[:, 0], points[:, 1], s=1, c=points[:, 2], cmap="viridis")
ax1.set_title("Top-down view (X vs Y) — click here to pick points")
ax1.set_xlabel("X")
ax1.set_ylabel("Y")
ax1.set_aspect("equal")

# Side view (X vs Z), for reference only
ax2.scatter(points[:, 0], points[:, 2], s=1, c=points[:, 1], cmap="viridis")
ax2.set_title("Side view (X vs Z) — reference only")
ax2.set_xlabel("X")
ax2.set_ylabel("Z")
ax2.set_aspect("equal")

picked = []

def on_click(event):
    if event.inaxes != ax1:
        return
    if event.xdata is None or event.ydata is None:
        return
    click_xy = np.array([event.xdata, event.ydata])
    dists = np.linalg.norm(points[:, :2] - click_xy, axis=1)
    idx = np.argmin(dists)
    pt = points[idx]
    picked.append(tuple(pt))
    print(f"Picked point #{len(picked)}: ({pt[0]:.5f}, {pt[1]:.5f}, {pt[2]:.5f})")
    ax1.plot(pt[0], pt[1], "r+", markersize=15, markeredgewidth=2)
    fig.canvas.draw()

fig.canvas.mpl_connect("button_press_event", on_click)

print("\nClick points in the LEFT plot. Close the window when done.")
print("Coordinates print here in the console as you click.\n")

plt.tight_layout()
plt.show()

print("\n" + "="*60)
print(f"All {len(picked)} picked points:")
for i, pt in enumerate(picked, 1):
    print(f"  {i}: ({pt[0]:.5f}, {pt[1]:.5f}, {pt[2]:.5f})")
print("="*60)
print("\nCopy these into MANUAL_CORNERS or MANUAL_CROP_CORNERS as needed.")
