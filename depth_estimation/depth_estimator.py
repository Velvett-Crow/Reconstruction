"""
Born out of Depth Anything V1 Issue 36
Make sure you have the necessary libraries installed.
Code by @1ssb

This script processes a set of images to generate depth maps and corresponding point clouds.
The resulting point clouds are saved in the specified output directory.

Usage:
    python script.py --encoder vitl --load-from path_to_model --max-depth 20 --img-path path_to_images --outdir output_directory --focal-length-x 470.4 --focal-length-y 470.4

Arguments:
    --encoder: Model encoder to use. Choices are ['vits', 'vitb', 'vitl', 'vitg'].
    --load-from: Path to the pre-trained model weights.
    --max-depth: Maximum depth value for the depth map.
    --img-path: Path to the input image or directory containing images.
    --outdir: Directory to save the output point clouds.
    --focal-length-x: Focal length along the x-axis.
    --focal-length-y: Focal length along the y-axis.
"""

import argparse
import cv2
import glob
import numpy as np
import open3d as o3d
import os
from PIL import Image
import torch
import sys

sys.path.append ('/home/jovyan/Depth-Anything-V2/metric_depth')

from depth_anything_v2.dpt import DepthAnythingV2

# DAv2_training/finetuned_vitb_endoslam_ddp_try2/best_endoslam.pth (path to fine-tuned model)
# metric_depth/depth_anything_v2_metric_hypersim_vitb.pth (path to pre-trained model)

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Generate depth maps and point clouds from images.')
    parser.add_argument('--encoder', default='vitb', type=str, choices=['vits', 'vitb', 'vitl', 'vitg'],
                        help='Model encoder to use.')
    parser.add_argument('--load-from', default='/home/jovyan/metric_depth/depth_anything_v2_metric_hypersim_vitb.pth', type=str,
                        required=False,
                        help='Path to the pre-trained model weights.')
    parser.add_argument('--max-depth', default=0.1, type=float,
                        help='Maximum depth value for the depth map.')
    parser.add_argument('--img-path', default='/home/jovyan/videos_renders/frames/dy/resized/rectified',
                        type=str, required=False,
                        help='Path to the input image or directory containing images.')
    parser.add_argument('--outdir', type=str, default='/home/jovyan/videos_renders/point_clouds/dy/resized/rectified',
                        help='Directory to save the output point clouds.')
    parser.add_argument('--focal-length-x', default=156.0418, type=float,
                        help='Focal length along the x-axis.')
    parser.add_argument('--focal-length-y', default=155.7529, type=float,
                        help='Focal length along the y-axis.')

    args = parser.parse_args()

    # Determine the device to use (CUDA, MPS, or CPU)
    DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

    # Model configuration based on the chosen encoder
    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }

    # Initialize the DepthAnythingV2 model with the specified configuration
    depth_anything = DepthAnythingV2(**{**model_configs[args.encoder], 'max_depth': args.max_depth})
    ckpt = torch.load(args.load_from, map_location='cpu')
    if 'model' in ckpt:
        ckpt = ckpt['model'] #extracting model weights
    depth_anything.load_state_dict(ckpt)
    depth_anything = depth_anything.to(DEVICE).eval()

    # Get the list of image files to process
    if os.path.isfile(args.img_path):
        if args.img_path.endswith('txt'):
            with open(args.img_path, 'r') as f:
                filenames = f.read().splitlines()
        else:
            filenames = [args.img_path]
    else:
        filenames = glob.glob(os.path.join(args.img_path, '**/*'), recursive=True)

    # Create the output directory if it doesn't exist
    os.makedirs(args.outdir, exist_ok=True)

    # Process each image file
    for k, filename in enumerate(filenames):
        # print(f'Processing {k+1}/{len(filenames)}: {filename}')

        # Load the image
        color_image = Image.open(filename).convert('RGB')
        width, height = color_image.size

        # Read the image using OpenCV
        image = cv2.imread(filename)
        pred = depth_anything.infer_image(image, height)

        # Resize depth prediction to match the original image size
        resized_pred = Image.fromarray(pred).resize((width, height), Image.NEAREST)
        
        # save depthmap
        # depth_npy_path = os.path.join(
        #    args.outdir,
        #    os.path.basename(filename).replace('.png', '_pred.npy')
        # )

        # np.save(depth_npy_path, pred_resized.astype(np.float32))

        # Generate mesh grid and calculate point cloud coordinates
        c_x = 157.13 - 40 # subtract by 40 when using data from MicroCam (cropped images)
        c_y = 168.92 - 40
        x, y = np.meshgrid(np.arange(width), np.arange(height))
        x = (x - c_x) / args.focal_length_x
        y = (y - c_y) / args.focal_length_y
        z = np.array(resized_pred)
        points = np.stack((np.multiply(x, z), np.multiply(y, z), z), axis=-1).reshape(-1, 3)
        colors = np.array(color_image).reshape(-1, 3) / 255.0

        # Create the point cloud and save it to the output directory
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        o3d.io.write_point_cloud(os.path.join(args.outdir, os.path.splitext(os.path.basename(filename))[0] + ".ply"), pcd,         write_ascii=True)

    print(f"Complete")

if __name__ == '__main__':
    main()