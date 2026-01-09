'''
Script for generating depth maps used during fine-tuning assessment
'''

import os
import cv2
import numpy as np
import torch
from PIL import Image
import open3d as o3d
import sys

sys.path.append('/home/jovyan/Depth-Anything-V2/metric_depth')

from depth_anything_v2.dpt import DepthAnythingV2

encoder = 'vitb'
weights_path = '/home/jovyan/metric_depth/depth_anything_v2_metric_hypersim_vitb.pth'
text_file = '/home/jovyan/Depth-Anything-V2/metric_depth/dataset/splits/C3VDv2/test.txt'

max_depth = 0.1
output_depth_dir = '/home/jovyan/videos_renders/depth_maps/C3VDv2/test_txt0/np_array'
output_pcd_dir = '/home/jovyan/videos_renders/point_clouds/C3VDv2/test_txt0'

focal_x = 156.0418
focal_y = 155.7529

# Read RGB image paths from the text file
# We take only the first column.

with open(text_file, 'r') as f:
    rgb_files = [line.split()[0] for line in f.read().splitlines()]

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

model_configs = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
}

model = DepthAnythingV2(**{**model_configs[encoder], 'max_depth': max_depth})
model.load_state_dict(torch.load(weights_path, map_location='cpu'))
model = model.to(DEVICE).eval()

os.makedirs(output_depth_dir, exist_ok=True)
os.makedirs(output_pcd_dir, exist_ok=True)

for idx, rgb_path in enumerate(rgb_files):
    print(f"Processing {idx+1}/{len(rgb_files)}: {rgb_path}")

    # Load image once (use OpenCV since model expects it)
    img_cv = cv2.imread(rgb_path)
    if img_cv is None:
        print(f"Error loading image: {rgb_path}")
        continue
    
    height, width = img_cv.shape[:2]
    
    # Get depth prediction
    depth_result = model.infer_image(img_cv, height)
        
        # Extract depth array
        depth_array = depth_result[0]
    else:
        depth_array = depth_result
    
    # Convert to numpy array
    if not isinstance(depth_array, np.ndarray):
        depth_array = np.array(depth_array)
    
    # Resize depth to match original image dimensions
    depth_np = cv2.resize(depth_array, (width, height), interpolation=cv2.INTER_NEAREST)
    
    # Save depth map as numpy file
    depth_npy_path = os.path.join(output_depth_dir, os.path.basename(rgb_path).replace('.png', '_pred.npy'))
    np.save(depth_npy_path, depth_np)
    
    # Also save as PNG for visualization
    # depth_normalized = (depth_np / depth_np.max() * 65535).astype(np.uint16)  # Use 16-bit for better precision
    # depth_png_path = os.path.join(output_depth_dir, os.path.basename(rgb_path).replace('.png', '_pred.png'))
    # cv2.imwrite(depth_png_path, depth_normalized)
    
    # Load original image for colors
    img = Image.open(rgb_path).convert("RGB")