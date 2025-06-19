import os
import sys
import cv2
from tqdm import tqdm

# ======== 全局路径配置 ========
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))  # 当前目录：/workspace/semantic_module
IMAGE_DIR = "/workspace/dataset/gerrard-hall-test2/images/"
SCENE_NAME = os.path.basename(os.path.dirname(IMAGE_DIR.rstrip("/")))

OUTPUT_BASE = os.path.join(PROJECT_ROOT, "output", "masks_png", SCENE_NAME)
OUTPUT_INSTANCE_DIR = os.path.join(OUTPUT_BASE, "instance")
OUTPUT_SEMANTIC_DIR = os.path.join(OUTPUT_BASE, "semantic")

# ======== Wrapper 导入 ========
sys.path.insert(0, PROJECT_ROOT)
from sam2_wrapper import sam2_generate_instance_mask
from seem_wrapper import generate_mask_instance

# ======== 初始化文件夹 ========
os.makedirs(OUTPUT_INSTANCE_DIR, exist_ok=True)
os.makedirs(OUTPUT_SEMANTIC_DIR, exist_ok=True)
valid_exts = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}

image_list = sorted([
    f for f in os.listdir(IMAGE_DIR)
    if os.path.splitext(f)[-1] in valid_exts
])

print(f"📂 Scene: {SCENE_NAME}")
print(f"📁 Input Dir: {IMAGE_DIR}")
print(f"📁 Output Dir: {OUTPUT_BASE}")
print(f"🖼️ Total images found: {len(image_list)}")

# ======== Stage 1: Semantic Mask ========
print("\n🔷 Stage 1: Generating Semantic Masks (SEEM)")
for fname in tqdm(image_list, desc="Semantic"):
    try:
        path = os.path.join(IMAGE_DIR, fname)
        image = cv2.imread(path)
        if image is None:
            print(f"[WARN] Cannot read image: {path}")
            continue
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = generate_mask_instance(rgb)

        out_path = os.path.join(OUTPUT_SEMANTIC_DIR, os.path.splitext(fname)[0] + "_semantic.png")
        cv2.imwrite(out_path, mask)
    except Exception as e:
        print(f"[ERROR] Semantic mask failed: {fname} ({e})")

# ======== Stage 2: Instance Mask ========
print("\n🔶 Stage 2: Generating Instance Masks (SAM2)")
for fname in tqdm(image_list, desc="Instance"):
    try:
        path = os.path.join(IMAGE_DIR, fname)
        image = cv2.imread(path)
        if image is None:
            print(f"[WARN] Cannot read image: {path}")
            continue
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = sam2_generate_instance_mask(rgb)

        out_path = os.path.join(OUTPUT_INSTANCE_DIR, os.path.splitext(fname)[0] + "_instance.png")
        cv2.imwrite(out_path, mask)
    except Exception as e:
        print(f"[ERROR] Instance mask failed: {fname} ({e})")

print(f"\n✅ All masks generated successfully for scene: {SCENE_NAME}")
