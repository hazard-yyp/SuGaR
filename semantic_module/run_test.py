import sys
import os
import cv2

# 添加当前目录到 sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 导入两个 wrapper
from seem_wrapper import generate_mask_instance as generate_semantic_mask
from sam2_wrapper import sam2_generate_instance_mask as generate_instance_mask

# 输入图像路径
image_path = '/workspace/semantic_module/IMG_2331.JPG'

# 输出路径
output_dir = '/workspace/semantic_module/output/'
semantic_path = os.path.join(output_dir, 'masks_semantic_png', 'IMG_2331_semantic.png')
instance_path = os.path.join(output_dir, 'masks_instance_png', 'IMG_2331_instance.png')

# 读取图像
image_np = cv2.imread(image_path)
if image_np is None:
    raise ValueError(f"Failed to read image: {image_path}")
image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)

# 生成 semantic mask（SEEM）
semantic_mask = generate_semantic_mask(image_np)
os.makedirs(os.path.dirname(semantic_path), exist_ok=True)
cv2.imwrite(semantic_path, semantic_mask)
print(f"✅ Semantic mask saved to {semantic_path}")

# 生成 instance mask（SAM2）
instance_mask = generate_instance_mask(image_np)
os.makedirs(os.path.dirname(instance_path), exist_ok=True)
cv2.imwrite(instance_path, instance_mask)
print(f"✅ Instance mask saved to {instance_path}")