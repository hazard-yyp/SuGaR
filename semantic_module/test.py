# test.py

import cv2
import numpy as np
import sys
import os

# 添加 semantic_module 到 sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from seem_wrapper import generate_mask
import os

# 输入一张测试图像（你改成你的一张测试图路径）
test_image_path = '/workspace/cosplat/semantic_module/IMG_2331.JPG'

# 输出 mask 存放路径
output_mask_path = '/workspace/cosplat/semantic_module/output/masks_png/IMG_2331.png'

# 读取测试图像
image_np = cv2.imread(test_image_path)
if image_np is None:
    raise ValueError(f"Failed to read image: {test_image_path}")
image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)

# 生成 mask
mask = generate_mask(image_np)

# 确保输出目录存在
os.makedirs(os.path.dirname(output_mask_path), exist_ok=True)

# 保存 mask
cv2.imwrite(output_mask_path, mask)

print(f"Mask saved to {output_mask_path}")