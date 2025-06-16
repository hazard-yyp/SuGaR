import os
import sys
import torch
import numpy as np
import cv2
from hydra import initialize_config_dir, compose
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf
import scipy.ndimage as ndimage  # morphology filtering

# ======== Adjust local path for SAM2 source ========
this_dir = os.path.dirname(os.path.abspath(__file__))
sam2_root = os.path.join(this_dir, "sam2")
sys.path.insert(0, sam2_root)

# ======== SAM2-specific imports ========
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.build_sam import build_sam2

# ======== SAM2 Config ========
SAM_CHECKPOINT = os.path.join(this_dir, "checkpoints", "sam2.1_hiera_large.pt")
SAM_CONFIG = "sam2.1_hiera_l.yaml"
SAM_CONFIG_DIR = os.path.join(sam2_root, "sam2", "configs", "sam2.1")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ======== Load SAM2 Model (Singleton style, only once) ========
if GlobalHydra.instance().is_initialized():
    GlobalHydra.instance().clear()

with initialize_config_dir(config_dir=SAM_CONFIG_DIR, job_name="sam2_job"):
    cfg = compose(config_name=SAM_CONFIG)
    OmegaConf.resolve(cfg)

model = build_sam2(
    config_file=SAM_CONFIG,
    ckpt_path=SAM_CHECKPOINT,
    device=DEVICE,
    hydra_config_dir=SAM_CONFIG_DIR,
)
sam2_predictor = SAM2ImagePredictor(model)
print("✅ SAM2 instance segmentation model loaded (Prompt-Guided BatchSafe FINAL_FIX3 INDUSTRY).")

# ======== Grid Box Generator ========
def generate_grid_boxes(image_shape, grid_size=16, box_size_ratio=0.1):
    """
    Generate grid of box prompts.

    image_shape: (H,W,3)
    grid_size: number of grid cells per dimension
    box_size_ratio: box size as a fraction of image size
    """
    H, W = image_shape[:2]
    box_size_x = W * box_size_ratio
    box_size_y = H * box_size_ratio

    step_x = W / grid_size
    step_y = H / grid_size

    boxes = []
    for i in range(grid_size):
        for j in range(grid_size):
            center_x = (i + 0.5) * step_x
            center_y = (j + 0.5) * step_y

            x1 = max(0, center_x - box_size_x / 2)
            y1 = max(0, center_y - box_size_y / 2)
            x2 = min(W, center_x + box_size_x / 2)
            y2 = min(H, center_y + box_size_y / 2)

            boxes.append([x1, y1, x2, y2])

    return np.array(boxes, dtype=np.float32)

# ======== Public API: Prompt-Guided Instance Mask ========
@torch.no_grad()
def sam2_generate_instance_mask(image_np, verbose=True, max_instance_id=None):
    """
    输入：image_np (H,W,3) RGB uint8
    输出：instance_mask (H,W) uint8，每个值为 instance_id，从 1 开始，0 为背景
    可选参数：
        verbose: 是否输出详细信息
        max_instance_id: 限制最大 instance 数量，None 表示不限制
    """

    image_rgb = image_np.copy()
    boxes = generate_grid_boxes(image_rgb.shape, grid_size=16, box_size_ratio=0.1)

    if verbose:
        print(f"Generated {len(boxes)} grid boxes.")
        print(f"Using batch_size=32 for Prompt-Guided predict.")

    batch_size = 32
    min_area_threshold = 1500
    morph_kernel = np.ones((5, 5))  # morphology opening kernel

    instance_mask = np.zeros((image_rgb.shape[0], image_rgb.shape[1]), dtype=np.uint8)
    instance_id = 1

    sam2_predictor.set_image(image_rgb)

    for i in range(0, len(boxes), batch_size):
        batch_boxes = boxes[i:i + batch_size]

        with torch.autocast("cuda" if torch.cuda.is_available() else "cpu", dtype=torch.bfloat16):
            masks, scores, logits = sam2_predictor.predict(
                point_coords=None,
                point_labels=None,
                box=batch_boxes,
            )

        masks_np = masks.cpu().numpy() if isinstance(masks, torch.Tensor) else masks

        for j in range(masks_np.shape[0]):
            if max_instance_id is not None and instance_id > max_instance_id:
                break

            m = masks_np[j]
            while m.ndim > 2:
                m = m[0]

            segmentation = m > 0.5
            segmentation = ndimage.binary_opening(segmentation, structure=morph_kernel).astype(bool)
            area = int(np.sum(segmentation))

            if area < min_area_threshold:
                continue

            instance_mask[segmentation] = instance_id
            instance_id += 1

    if verbose:
        print(f"✅ Total {instance_id - 1} valid instances written to mask.")

    return instance_mask
