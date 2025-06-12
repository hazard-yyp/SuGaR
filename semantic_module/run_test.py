# run_test.py

import sys
import os

# 添加当前目录到 sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from seem_wrapper import generate_mask
import cv2

# 输入一张测试图
image_path = '/workspace/semantic_module/IMG_2331.JPG'
output_path = '/workspace/semantic_module/output/masks_png/IMG_2331.png'

# 读取图像
image_np = cv2.imread(image_path)
if image_np is None:
    raise ValueError(f"Failed to read image: {image_path}")

image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)

# 生成 mask
mask = generate_mask(image_np)

# 保存
os.makedirs(os.path.dirname(output_path), exist_ok=True)
cv2.imwrite(output_path, mask)

print(f"Mask saved to {output_path}")
