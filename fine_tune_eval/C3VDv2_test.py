"""
Depth Evaluation Script for C3VDv2
---------------------------------
- Reads text file with RGB and GT depth paths
- Loads predicted depth from .npy files (<id>_pred.npy)
- Computes depth metrics for all images
- Calculates and saves mean metrics to .txt file
"""

import os
import cv2
import numpy as np
from PIL import Image

def load_gt_depth(path):
    """Load C3VDv2 TIFF depth (0-65535 → 0-100mm → meters)"""
    img = np.array(Image.open(path)).astype(np.float32)
    depth_mm = (img / 65535.0) * 100.0  # GT clamped at 100mm
    return depth_mm / 1000  # convert mm to meters

def load_pred_depth(path):
    """Load predicted depth from .npy file"""
    return np.load(path).astype(np.float32)

def depth_metrics(gt, pred):
    """Compute depth evaluation metrics"""
    eps = 1e-6
    mask = gt > eps  # ignore zero-depth GT pixels

    gt = gt[mask]
    pred = pred[mask] + eps

    ratio = np.maximum(gt / pred, pred / gt)

    return {
        "abs_rel": np.mean(np.abs(gt - pred) / gt),
        "sq_rel": np.mean(((gt - pred) ** 2) / gt),
        "rmse": np.sqrt(np.mean((gt - pred) ** 2)),
        "rmse_log": np.sqrt(np.mean((np.log(gt) - np.log(pred)) ** 2)),
        "d1": np.mean(ratio < 1.25),
        "silog": np.sqrt(
            np.mean((np.log(pred) - np.log(gt)) ** 2) -
            (np.mean(np.log(pred) - np.log(gt))) ** 2
        ),
    }

def evaluate_c3vdv2(list_file, pred_dir):
    """
    Evaluate depth predictions for C3VDv2 dataset
    
    Args:
        list_file: txt file with format: rgb_path gt_depth_path
        pred_dir: directory containing predicted depth .npy files
    """
    
    # Read the list file
    with open(list_file, "r") as f:
        rows = [line.split() for line in f.read().splitlines()]
    
    gt_paths = [row[1] for row in rows]  # GT depth path is 2nd column
    results = []
    
    print(f"Found {len(gt_paths)} GT depth maps.\n")
    
    for gt_path in gt_paths:
        # Extract base name from GT path
        base = os.path.basename(gt_path).replace("_depth.tiff", "")
        
        # Construct prediction filename
        pred_file = f"{base}_pred.npy"
        pred_path = os.path.join(pred_dir, pred_file)
        
        # Load depth maps
        gt = load_gt_depth(gt_path)
        pred = load_pred_depth(pred_path)
        
        # Compute metrics
        m = depth_metrics(gt, pred)
        m["name"] = base
        results.append(m)
        
    return results

if __name__ == "__main__":
    # Configuration
    list_file = "/home/jovyan/Depth-Anything-V2/metric_depth/dataset/splits/C3VDv2/test.txt"
    pred_dir = "/home/jovyan/videos_renders/depth_maps/C3VDv2/test_txt/np_array"
    
    # Evaluate
    results = evaluate_c3vdv2(list_file, pred_dir)
    
    # Print individual results
    print("\n" + "="*60)
    print("INDIVIDUAL RESULTS:")
    print("="*60)
    for r in results:
        print(f"\nImage {r['name']}:")
        print(f"  abs_rel:  {r['abs_rel']:.6f}")
        print(f"  sq_rel:   {r['sq_rel']:.6f}")
        print(f"  rmse:     {r['rmse']:.6f}")
        print(f"  rmse_log: {r['rmse_log']:.6f}")
        print(f"  silog:    {r['silog']:.6f}")
        print(f"  d1:       {r['d1']:.6f}")
    
    # Save results to text file
    output_txt = os.path.join(pred_dir, "evaluation_results.txt")
    
    with open(output_txt, "w") as f:
        f.write("### DEPTH EVALUATION RESULTS (C3VDv2) ###\n\n")
        
        # Write per-image metrics
        f.write("### PER-IMAGE METRICS ###\n\n")
        for r in results:
            f.write(f"Image {r['name']}:\n")
            f.write(f"  abs_rel:  {r['abs_rel']:.6f}\n")
            f.write(f"  sq_rel:   {r['sq_rel']:.6f}\n")
            f.write(f"  rmse:     {r['rmse']:.6f}\n")
            f.write(f"  rmse_log: {r['rmse_log']:.6f}\n")
            f.write(f"  silog:    {r['silog']:.6f}\n")
            f.write(f"  d1:       {r['d1']:.6f}\n\n")
        
        # Compute and write mean metrics
        if len(results) > 0:
            f.write("### MEAN METRICS ###\n\n")
            
            mean_abs_rel  = np.mean([r['abs_rel'] for r in results])
            mean_sq_rel   = np.mean([r['sq_rel'] for r in results])
            mean_rmse     = np.mean([r['rmse'] for r in results])
            mean_rmse_log = np.mean([r['rmse_log'] for r in results])
            mean_silog    = np.mean([r['silog'] for r in results])
            mean_d1       = np.mean([r['d1'] for r in results])

            f.write(f"Mean abs_rel:  {mean_abs_rel:.6f}\n")
            f.write(f"Mean sq_rel:   {mean_sq_rel:.6f}\n")
            f.write(f"Mean rmse:     {mean_rmse:.6f}\n")
            f.write(f"Mean rmse_log: {mean_rmse_log:.6f}\n")
            f.write(f"Mean silog:    {mean_silog:.6f}\n")
            f.write(f"Mean d1:       {mean_d1:.6f}\n")

    print(f"\nSaved evaluation results to:\n{output_txt}")