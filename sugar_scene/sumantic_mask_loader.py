import os
import numpy as np
import torch
from PIL import Image

class SemanticMaskLoader:
    """
    Loads precomputed binary masks (from SAM2) for each image.
    Each mask should be a .png file with the same basename as the input image.
    """
    def __init__(self, mask_dir, image_exts=['.jpg', '.jpeg', '.png']):
        self.mask_dir = os.path.abspath(mask_dir)
        self.image_exts = image_exts
        self.mask_map = {}

        if not os.path.exists(self.mask_dir):
            raise FileNotFoundError(f"Semantic mask directory not found: {self.mask_dir}")

        all_files = os.listdir(self.mask_dir)
        for file in all_files:
            if file.endswith("_mask.png"):
                basename = file.replace("_mask.png", "")
                self.mask_map[basename] = os.path.join(self.mask_dir, file)

    def get_mask_tensor(self, image_filename, height, width):
        """
        Given an image file name, load its corresponding semantic mask and return a [H, W] binary tensor.
        If no mask is found, return None.
        """
        basename = os.path.splitext(os.path.basename(image_filename))[0]
        if basename not in self.mask_map:
            return None
        
        mask_path = self.mask_map[basename]
        mask_img = Image.open(mask_path).convert("L").resize((width, height), Image.NEAREST)
        mask_np = np.array(mask_img)
        mask_tensor = torch.from_numpy(mask_np > 0).bool()  # binary mask
        return mask_tensor
