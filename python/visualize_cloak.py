import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
import os
import random

def visualize_mobile_cloak(image_path, npy_path, alpha=0.7):
    visualize_mobile_cloak_batch([image_path], npy_path, alpha=alpha)


def visualize_mobile_cloak_batch(image_paths, npy_path, alpha=0.7):
    # Load the UAP once (.npy file)
    # Shape is (1, 3, 224, 224), we need (224, 224, 3)
    uap = np.load(npy_path)
    uap = np.transpose(uap.squeeze(), (1, 2, 0))

    num_images = len(image_paths)
    if num_images == 0:
        print("No images provided for visualization.")
        return

    fig, axes = plt.subplots(num_images, 3, figsize=(14, 4 * num_images))
    if num_images == 1:
        axes = np.expand_dims(axes, axis=0)

    for row, image_path in enumerate(image_paths):
        # 1. Load Original Image
        original_img = Image.open(image_path).convert('RGB')
        original_img = original_img.resize((224, 224))  # Match UAP size
        original_arr = np.array(original_img) / 255.0

        # 2. Apply Alpha Blending Math
        # Image_new = (1 - alpha) * Original + alpha * (Original + UAP)
        # This matches your UniversalPerturbationGenerator logic
        cloaked_arr = (1 - alpha) * original_arr + alpha * np.clip(original_arr + uap, 0, 1)
        cloaked_arr = np.clip(cloaked_arr, 0, 1)

        axes[row, 0].imshow(original_arr)
        if row == 0:
            axes[row, 0].set_title("Original Image (Unprotected)", fontsize=11, pad=10)
        axes[row, 0].axis('off')

        # Visualization of the UAP alone (scaled up for visibility)
        axes[row, 1].imshow(np.clip(uap * 10 + 0.5, 0, 1))
        if row == 0:
            axes[row, 1].set_title("The 'Cloak' (UAP) Pattern", fontsize=11, pad=10)
        axes[row, 1].axis('off')

        axes[row, 2].imshow(cloaked_arr)
        if row == 0:
            axes[row, 2].set_title(f"Cloaked Image (Alpha={alpha})", fontsize=11, pad=10)
        axes[row, 2].axis('off')

    plt.tight_layout(pad=3.0)
    plt.show()

if __name__ == "__main__":
    # Test on a COCO image
    import sys
    
    # Get the script directory and construct absolute paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    workspace_root = os.path.dirname(script_dir)
    coco_dir = os.path.join(workspace_root, "data", "MS-COCO", "val2017")
    uap_path = os.path.join(workspace_root, "data", "results", "clip_uap_final.npy")
    
    # Check if directories exist
    if not os.path.exists(coco_dir):
        print(f"ERROR: COCO directory not found: {coco_dir}")
        print("Please ensure MS-COCO dataset is downloaded.")
        sys.exit(1)
    
    if not os.path.exists(uap_path):
        print(f"ERROR: UAP file not found: {uap_path}")
        print("Please run clip_uap_generator.py first to generate the UAP.")
        sys.exit(1)
    
    # Find sample images from COCO
    sample_images = [f for f in os.listdir(coco_dir) if f.endswith('.jpg')]
    
    if not sample_images:
        print(f"ERROR: No COCO images found in {coco_dir}")
        sys.exit(1)
    
    sample_images = random.sample(sample_images, k=min(5, len(sample_images)))
    sample_paths = [os.path.join(coco_dir, f) for f in sample_images]
    print(f"Using {len(sample_paths)} sample images:")
    for name in sample_images:
        print(f"  - {name}")

    visualize_mobile_cloak_batch(
        image_paths=sample_paths,
        npy_path=uap_path,
        alpha=0.7
    )