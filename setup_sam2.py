import os
import subprocess
import sys
import requests

# ======== Setup paths and config ========
SAM2_CHECKPOINTS_DIR = os.path.join("semantic_module", "checkpoints")
SAM2_CHECKPOINT_PATH = os.path.join(SAM2_CHECKPOINTS_DIR, "sam_vit_h_6b5553.pth")
SAM2_CHECKPOINT_URL = "https://dl.fbaipublicfiles.com/sam/sam_vit_h_6b5553.pth"

# ======== Install dependencies if missing ========
def ensure_package(pkg_name):
    try:
        __import__(pkg_name)
    except ImportError:
        print(f"Installing missing dependency: {pkg_name}...")
        subprocess.run([sys.executable, "-m", "pip", "install", pkg_name], check=True)

for pkg in ["hydra-core", "omegaconf", "timm", "einops", "opencv-python", "tqdm", "Pillow"]:
    ensure_package(pkg)

# ======== Download SAM2 checkpoint if missing ========
os.makedirs(SAM2_CHECKPOINTS_DIR, exist_ok=True)
if not os.path.exists(SAM2_CHECKPOINT_PATH):
    print(f"Downloading SAM2 checkpoint to {SAM2_CHECKPOINT_PATH}...")
    response = requests.get(SAM2_CHECKPOINT_URL, stream=True)
    with open(SAM2_CHECKPOINT_PATH, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    print("✅ Checkpoint downloaded.")
else:
    print("✅ Checkpoint already exists.")

print("✅ SAM2 setup complete. Ready for inference.")
