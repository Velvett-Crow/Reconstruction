import numpy as np
import cv2
import sys
import os

sys.path.append ('edge filter/sam2')

from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

def setup_sam_model(checkpoint_path, device='cuda'):
    model_cfg = 'sam2_hiera_b+.yaml'
    predictor = SAM2ImagePredictor(build_sam2(model_cfg, checkpoint_path, device=device, apply_postprocessing=False))
    
    return predictor
    # Initialize SAM model

def create_guidance_image_from_sam(image, sam_model):
    
    #Create guidance image from SAM segmentation masks
    
    mask_generator = SAM2AutomaticMaskGenerator(sam_model)
    masks = mask_generator.generate(image)
    
    height, width = image.shape[:2]
    edge_map = np.zeros((height, width), dtype=np.uint8)
    
    for mask in masks:
        segmentation = mask['segmentation']
        contours, _ = cv2.findContours(
            segmentation.astype(np.uint8), 
            cv2.RETR_EXTERNAL, 
            cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(edge_map, contours, -1, 255, thickness=2)
    
    guidance_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    kernel = np.ones((3, 3), np.uint8)
    dilated_edges = cv2.dilate(edge_map, kernel, iterations=1)
    guidance_image[dilated_edges > 0] = 0
    
    return guidance_image.astype(np.float32), edge_map