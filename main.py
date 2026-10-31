import os
import subprocess

colmap_dir = r"C:\Users\ASUS\Downloads\colmap-x64-windows-cuda\bin"
colmap_path = os.path.join(colmap_dir, "colmap.exe")

env = os.environ.copy()
env["QT_QPA_PLATFORM"] = "offscreen"
env["QT_QPA_PLATFORM_PLUGIN_PATH"] = os.path.join(colmap_dir, "platforms")

def run(args):
    subprocess.run(
        [colmap_path] + args,
        check=True,
        cwd=colmap_dir,
        env=env
    )

# -------------------------
# 1. Feature extraction
# -------------------------
run([
    "feature_extractor",
    "--database_path", r"D:\MinProject\db.db",
    "--image_path", r"D:\MinProject\images",
    "--ImageReader.single_camera", "1"
])

# -------------------------
# 2. Matching
# -------------------------
run([
    "sequential_matcher",
    "--database_path", r"D:\MinProject\db.db"
])

# -------------------------
# 3. Sparse reconstruction
# -------------------------
os.makedirs(r"D:\MinProject\sparse", exist_ok=True)

run([
    "mapper",
    "--database_path", r"D:\MinProject\db.db",
    "--image_path", r"D:\MinProject\images",
    "--output_path", r"D:\MinProject\sparse"
])

# -------------------------
# 4. Dense reconstruction 🔥 (ADD HERE)
# -------------------------
os.makedirs(r"D:\MinProject\dense", exist_ok=True)

run([
    "image_undistorter",
    "--image_path", r"D:\MinProject\images",
    "--input_path", r"D:\MinProject\sparse\0",
    "--output_path", r"D:\MinProject\dense",
    "--output_type", "COLMAP"
])

run([
    "patch_match_stereo",
    "--workspace_path", r"D:\MinProject\dense",
    "--workspace_format", "COLMAP",
    "--PatchMatchStereo.max_image_size", "1200",   # increase
    "--PatchMatchStereo.num_iterations", "5"
])

run([
    "stereo_fusion",
    "--workspace_path", r"D:\MinProject\dense",
    "--workspace_format", "COLMAP",
    "--output_path", r"D:\MinProject\dense\fused.ply"
])

run([
    "poisson_mesher",
    "--input_path", r"D:\MinProject\dense\fused.ply",
    "--output_path", r"D:\MinProject\dense\meshed.ply"
])

print("✅ FULL 3D Reconstruction Done")