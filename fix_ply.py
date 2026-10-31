import open3d as o3d

INPUT_PATH = r"D:\MinProject\dense\fused.ply"
OUTPUT_PATH = r"D:\MinProject\dense\fused_clean.ply"

pcd = o3d.io.read_point_cloud(INPUT_PATH)
print(f"Loaded {len(pcd.points)} points from {INPUT_PATH}")

# Re-write in a clean, standard PLY format (ASCII is most compatible;
# switch write_ascii=False for a smaller binary file once this works).
o3d.io.write_point_cloud(OUTPUT_PATH, pcd, write_ascii=True)
print(f"Saved clean copy to {OUTPUT_PATH}")
print("Open THIS file in CloudCompare instead of the original fused.ply")
