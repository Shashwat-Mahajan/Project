import open3d as o3d
import numpy as np

INPUT_PATH = r"D:\MinProject\dense\fused.ply"
OUTPUT_PATH = r"D:\MinProject\dense\fused.xyz"

pcd = o3d.io.read_point_cloud(INPUT_PATH)
points = np.asarray(pcd.points)
o3d.visualization.draw_geometries([pcd])
print(f"Loaded {len(points)} points from {INPUT_PATH}")
mesh = o3d.io.read_triangle_mesh(r"D:\MinProject\dense\meshed.ply")
mesh.compute_vertex_normals()
o3d.visualization.draw_geometries([mesh])

# Plain text, one point per line: X Y Z
# No header, no format ambiguity -- CloudCompare (and anything else)
# reads this trivially.
np.savetxt(OUTPUT_PATH, points, fmt="%.6f")
print(f"Saved {len(points)} points to {OUTPUT_PATH}")
print("Open THIS .xyz file in CloudCompare instead of the .ply")
