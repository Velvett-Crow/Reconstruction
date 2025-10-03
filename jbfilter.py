import numpy as np
import cv2

def load_depth_map(depth_path):
    
    """Load and normalize depth map"""
    
    depth = cv2.imread(depth_path, cv2.IMREAD_GRAYSCALE)
    return cv2.normalize(depth.astype(np.float32), None, 0, 1, cv2.NORM_MINMAX)

'''
def joint_bilateral_filter(depth_map, guidance_image, d, sigmaColor, sigmaSpace):
    """Apply Joint Bilateral Filter to depth map using guidance image"""
    # Your filter implementation from earlier
    H, W = depth_map.shape
    pad = sigmaSpace // 2
    depth_padded = cv2.copyMakeBorder(depth_map, pad, pad, pad, pad, cv2.BORDER_REFLECT)
    guidance_padded = cv2.copyMakeBorder(guidance_image, pad, pad, pad, pad, cv2.BORDER_REFLECT)
    
    # ... rest of your filter implementation ...
    
    return filtered_depth
'''

def refine(guide, raw_depth, d, sigmaColor, sigmaSpace):
    
    """Simpler version using OpenCV's implementation"""
    
    return cv2.ximgproc.jointBilateralFilter(
        guide, 
        raw_depth.astype(np.float32),
        d=d,
        sigmaColor=sigmaColor,
        sigmaSpace=sigmaSpace
    )