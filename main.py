import cv2
import numpy as np
import os
from SAM import setup_sam_model, create_guidance_image_from_sam
from jbfilter import load_depth_map, refine  #, joint_bilateral_filter
from visualize import visualize_comparison
from sam2.build_sam import build_sam2
import torch

# select the device for computation
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

if device.type == "cuda":
    # use bfloat16 for the entire notebook
    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
elif device.type == "mps":
    print(
        "\nSupport for MPS devices is preliminary. SAM 2 is trained with CUDA and might "
        "give numerically different outputs and sometimes degraded performance on MPS. "
        "See e.g. https://github.com/pytorch/pytorch/issues/84936 for a discussion."
    )


def main():
    # Initialize models
    
    sam2_checkpoint = '/home/jovyan/edge filter/sam2_hiera_base_plus.pt'
    model_cfg = 'sam2_hiera_b+.yaml'
    sam_model = build_sam2(model_cfg, sam2_checkpoint, device=device, apply_postprocessing=False)
    
    # Load your data    
    
    image = cv2.imread('/home/jovyan/Images/WIN_20220122_16_41_26_Pro.jpg')
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    raw_depth = load_depth_map('/home/jovyan/depth_map.png')
    
    # Create guidance from SAM
    guide, edge_map = create_guidance_image_from_sam(image_rgb, sam_model)
    
    # Apply refinement
    refined_depth = refine(
        raw_depth, 
        guide,
        d=15,
        sigmaColor=0.05, 
        sigmaSpace=25
    )
    
    # Or use the OpenCV version for faster processing
    # refined_depth = quick_opencv_refine(guidance_image, raw_depth)
    
    # Visualize results
    visualize_comparison(raw_depth, refined_depth, edge_map, guide)
    
    # Save result
    refined_uint8 = (refined_depth * 255).astype(np.uint8)
    cv2.imwrite('refined_depth.jpg', refined_uint8)

if __name__ == "__main__":
    main()