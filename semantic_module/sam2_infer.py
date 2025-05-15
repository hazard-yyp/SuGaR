import os
import sys
import json
import torch
import numpy as np
import cv2
from tqdm import tqdm
from PIL import Image
import requests

# ======== Adjust local path for SAM2 source ========
sam2_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "sam2"))
sys.path.insert(0, sam2_root)

# ======== SAM2-specific imports from local clone ========
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.build_sam import build_sam2
from hydra import initialize_config_dir, compose
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf

# ========== Config ==========
IMAGE_DIR = "/workspace/dataset/gerrard-hall-test/images/"
SCENE_NAME = os.path.basename(os.path.dirname(IMAGE_DIR.rstrip('/')))
OUTPUT_MASK_DIR = f"./semantic_module/output/masks_png/{SCENE_NAME}/"
OUTPUT_JSON_DIR = f"./semantic_module/output/masks_json/{SCENE_NAME}/"
SAM_CHECKPOINT = "./semantic_module/checkpoints/sam2.1_hiera_large.pt"
SAM_CONFIG = "sam2.1_hiera_l.yaml"
SAM_CONFIG_DIR = "./semantic_module/sam2/sam2/configs/sam2.1"
SAM_CHECKPOINT_URL = "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ========== Ensure output dirs exist ==========
os.makedirs(OUTPUT_MASK_DIR, exist_ok=True)
os.makedirs(OUTPUT_JSON_DIR, exist_ok=True)

# ========== Auto-download checkpoint if missing ==========
if not os.path.exists(SAM_CHECKPOINT):
    print(f"Checkpoint not found at {SAM_CHECKPOINT}, downloading from {SAM_CHECKPOINT_URL}...")
    response = requests.get(SAM_CHECKPOINT_URL, stream=True)
    with open(SAM_CHECKPOINT, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    print("Download complete.")

# ========== Load SAM2 Model ==========
print("Loading SAM2 model...")
if GlobalHydra.instance().is_initialized():
    GlobalHydra.instance().clear()

with initialize_config_dir(config_dir=os.path.abspath(SAM_CONFIG_DIR), job_name="sam2_job"):
    cfg = compose(config_name=SAM_CONFIG)
    OmegaConf.resolve(cfg)

model = build_sam2(
    config_file=SAM_CONFIG,
    ckpt_path=SAM_CHECKPOINT,
    device=DEVICE,
    hydra_config_dir=SAM_CONFIG_DIR,
)
sam_predictor = SAM2ImagePredictor(model)
print("✅ SAM2 loaded.")

# ========== Main processing loop ==========
all_images = sorted([f for f in os.listdir(IMAGE_DIR) if f.lower().endswith(('jpg', 'png', 'jpeg'))])

for image_name in tqdm(all_images, desc="Generating SAM2 masks"):
    image_id = os.path.splitext(image_name)[0]
    image_path = os.path.join(IMAGE_DIR, image_name)
    image_bgr = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        sam_predictor.set_image(image_rgb)
        masks, _, _ = sam_predictor.predict(
            point_coords=None,
            point_labels=None,
            box=None,
        )

    # Convert boolean masks to dict format
    masks_list = []
    for i in range(masks.shape[0]):
        m = masks[i]
        area = int(np.sum(m))
        masks_list.append({
            "segmentation": m,
            "area": area
        })

    # Sort masks by area descending
    masks_list = sorted(masks_list, key=lambda x: x['area'], reverse=True)

    # Save colored mask visualization
    mask_vis = np.zeros((image_rgb.shape[0], image_rgb.shape[1], 3), dtype=np.uint8)
    for i, mask in enumerate(masks_list):
        color = np.random.randint(0, 255, (3,), dtype=np.uint8)
        segmentation = np.array(mask['segmentation'], dtype=bool)  # 显式转为布尔类型
        mask_vis[segmentation] = color

    cv2.imwrite(os.path.join(OUTPUT_MASK_DIR, f"{image_id}_mask.png"), cv2.cvtColor(mask_vis, cv2.COLOR_RGB2BGR))

    # Save raw masks to JSON
    for m in masks_list:
        if isinstance(m['segmentation'], np.ndarray):
            m['segmentation'] = m['segmentation'].tolist()

    with open(os.path.join(OUTPUT_JSON_DIR, f"{image_id}_mask.json"), 'w') as f:
        json.dump(masks_list, f, indent=2)

print("✅ All SAM2 masks generated and saved.")
