# seem_wrapper.py
import sys
import os
import torch
import torchvision.transforms as T
import numpy as np
from PIL import Image
import torch.nn.functional as F

# 添加 seem_src 到 sys.path
this_dir = os.path.dirname(os.path.abspath(__file__))
seem_src_dir = os.path.join(this_dir, 'seem_src')
sys.path.insert(0, seem_src_dir)

# import SEEM 组件
from modeling.BaseModel import BaseModel
from modeling import build_model
from utils.arguments import load_opt_from_config_files
from utils.constants import COCO_PANOPTIC_CLASSES

# 配置路径
CHECKPOINT_PATH = os.path.join(seem_src_dir, 'checkpoints', 'xdecoder_focall_last_oq101.pt')
CONFIG_PATH = os.path.join(seem_src_dir, 'configs', 'seem', 'focall_unicl_lang_demo.yaml')

# load config
opt = load_opt_from_config_files([CONFIG_PATH])
opt['device'] = 'cuda'

# 加载模型
print("Loading SEEM model...")
model = BaseModel(opt, build_model(opt)).from_pretrained(CHECKPOINT_PATH).eval().cuda()

# 初始化 lang encoder
with torch.no_grad():
    model.model.sem_seg_head.predictor.lang_encoder.get_text_embeddings(COCO_PANOPTIC_CLASSES + ["background"], is_eval=True)
print("SEEM model loaded.")

# 定义 transform
transform = T.Compose([
    T.Resize((512, 512)),  # 可调整
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
])

# 通用内部函数
@torch.no_grad()
def _forward_mask(image_np, task_type='semantic'):
    image_pil = Image.fromarray(image_np)
    image_tensor = transform(image_pil).unsqueeze(0).cuda()

    orig_h, orig_w = image_np.shape[:2]

    with torch.autocast(device_type='cuda', dtype=torch.float16):
        features = model.model.backbone(image_tensor)
        results = model.model.sem_seg_head(features, None, None, None, task=task_type, extra={})

        pred_masks = results['pred_masks']  # [B, C, H_mask, W_mask]

        # Upsample to original image size
        pred_masks_upsampled = F.interpolate(
            pred_masks, size=(orig_h, orig_w), mode='bilinear', align_corners=False
        )

    pred_mask_np = pred_masks_upsampled.argmax(dim=1)[0].cpu().numpy().astype(np.uint8)
    return pred_mask_np

# Public API → 供外部调用
def generate_mask(image_np):
    """Semantic mask"""
    return _forward_mask(image_np, task_type='semantic')

def generate_mask_instance(image_np):
    """Instance mask"""
    return _forward_mask(image_np, task_type='instance')
