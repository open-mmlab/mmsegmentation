import os
import mmcv
import numpy as np
import cv2
from mmseg.apis import init_model, inference_model

# Configuration and checkpoint paths
CONFIG_FILE = '/Users/anyego/Desktop/mmsegmentationEQ2/configs/fcn/fcn_r18-d8_4xb2-80k_deepglobe-512x1024.py'
CHECKPOINT_FILE = '/Users/anyego/Desktop/mmsegmentationEQ2/workdir/fcn/iter_10000.pth'
DEVICE = 'cpu'

# Custom color palette
PALETTE = np.array([
    [0, 255, 255], [255, 255, 0], [255, 0, 255], 
    [0, 255, 0], [0, 0, 255], [255, 255, 255], 
    [0, 0, 0]
], dtype=np.uint8)

# Directories
IMAGE_DIR = '/Users/anyego/Desktop/mmsegmentationEQ2/Test_model'
OUTPUT_DIR = '/Users/anyego/Desktop/mmsegmentationEQ2/PrediccionesFCN'

def initialize_model(config_path, checkpoint_path, device):
    """Initialize the segmentation model."""
    return init_model(config_path, checkpoint_path, device=device)

def create_output_directory(directory):
    """Create the output directory if it doesn't exist."""
    if not os.path.exists(directory):
        os.makedirs(directory)

def get_image_paths(directory):
    """Retrieve all image paths from the directory."""
    return [os.path.join(directory, img_name) for img_name in os.listdir(directory)]

def postprocess_mask(mask):
    """Post-process the predicted mask."""
    # Apply morphological operations to refine the mask
    kernel = np.ones((3, 3), np.uint8)
    refined_mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return refined_mask

def process_image(model, img_path):
    """Run inference on a single image and return the color mask."""
    result = inference_model(model, img_path)
    mask = result.pred_sem_seg.data.cpu().numpy().astype(np.uint8).squeeze(0)
    mask = postprocess_mask(mask)
    return PALETTE[mask]

def save_color_mask(color_mask, output_path):
    """Save the color mask to the specified path."""
    mmcv.imwrite(color_mask, output_path)

def main():
    """Main function to run the segmentation inference."""
    model = initialize_model(CONFIG_FILE, CHECKPOINT_FILE, DEVICE)
    create_output_directory(OUTPUT_DIR)
    
    for img_path in get_image_paths(IMAGE_DIR):
        color_mask = process_image(model, img_path)
        if color_mask.size > 0:
            output_path = os.path.join(OUTPUT_DIR, os.path.basename(img_path))
            save_color_mask(color_mask, output_path)
        else:
            print(f"Skipping {img_path} due to invalid mask")

    print(f"Inference completed. Results saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()