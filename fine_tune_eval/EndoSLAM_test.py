'''Used during the evaluation of the fine-tuned model to the EndoSLAM test set'''

import os
import cv2
import numpy as np

def load_endoslam_gt_depth(path):
    depth = cv2.imread(path, cv2.IMREAD_UNCHANGED).astype(np.float32)

    # Any channel can be used here as they carry identical content
    if depth.ndim == 3:
        depth_mm = depth[:, :, 2]
    else:
        depth_mm = depth

    return depth_mm * 0.001   # convert mm to meters

def load_pred_depth(path):
    return np.load(path).astype(np.float32)

def depth_metrics(gt, pred):
    eps = 1e-6
    mask = gt > eps

    gt = gt[mask]
    pred = pred[mask] + eps

    ratio = np.maximum(gt / pred, pred / gt)

    return {
        "abs_rel": np.mean(np.abs(gt - pred) / gt),
        "sq_rel":  np.mean(((gt - pred) ** 2) / gt),
        "rmse":    np.sqrt(np.mean((gt - pred) ** 2)),
        "rmse_log": np.sqrt(np.mean((np.log(gt) - np.log(pred)) ** 2)),
        "silog": np.sqrt(
            np.mean((np.log(pred) - np.log(gt)) ** 2) -
            (np.mean(np.log(pred) - np.log(gt))) ** 2
        ),
        "d1": np.mean(ratio < 1.25),
        "d2": np.mean(ratio < 1.25**2),
        "d3": np.mean(ratio < 1.25**3),
    }

def evaluate_endoslam(gt_dir, pred_dir):

    gt_files = sorted([f for f in os.listdir(gt_dir) if f.endswith(".png")])
    results = []

    for gt_file in gt_files:

        # Extract numeric ID
        # aov_image_0201.png → 0201
        img_id = gt_file.replace("aov_image_", "").replace(".png", "")

        # Construct prediction filename
        pred_file = f"image_{img_id}_pred.npy"

        pred_path = os.path.join(pred_dir, pred_file)
        gt_path = os.path.join(gt_dir, gt_file)

        # Load depth maps
        gt = load_endoslam_gt_depth(gt_path)
        pred = load_pred_depth(pred_path)

        # Compute metrics
        m = depth_metrics(gt, pred)
        m["name"] = img_id
        results.append(m)

    return results

if __name__ == "__main__":

    gt_folder = "/home/jovyan/videos_renders/depth_maps/EndoSLAM/ground_truth"
    pred_folder = "/home/jovyan/videos_renders/depth_maps/EndoSLAM/depthmap_npy/fine_tuned"

    results = evaluate_endoslam(gt_folder, pred_folder)

    for r in results:
        print(r)

    # Save results into a text file
    output_txt = os.path.join(pred_folder, "evaluation_results_0.045.txt")

    with open(output_txt, "w") as f:

        f.write("### DEPTH EVALUATION RESULTS ###\n\n")

        # Write per-image metrics
        for r in results:
            f.write(f"Image {r['name']}:\n")
            f.write(f"  abs_rel:  {r['abs_rel']:.6f}\n")
            f.write(f"  sq_rel:   {r['sq_rel']:.6f}\n")
            f.write(f"  rmse:     {r['rmse']:.6f}\n")
            f.write(f"  rmse_log: {r['rmse_log']:.6f}\n")
            f.write(f"  silog:    {r['silog']:.6f}\n")
            f.write(f"  d1:       {r['d1']:.6f}\n")
            f.write(f"  d2:       {r['d2']:.6f}\n")
            f.write(f"  d3:       {r['d3']:.6f}\n\n")

        # Compute and write mean metrics
        if len(results) > 0:
            f.write("### MEAN METRICS ###\n")

            mean_abs_rel  = np.mean([r['abs_rel'] for r in results])
            mean_sq_rel   = np.mean([r['sq_rel'] for r in results])
            mean_rmse     = np.mean([r['rmse'] for r in results])
            mean_rmse_log = np.mean([r['rmse_log'] for r in results])
            mean_silog    = np.mean([r['silog'] for r in results])
            mean_d1       = np.mean([r['d1'] for r in results])
            mean_d2       = np.mean([r['d2'] for r in results])
            mean_d3       = np.mean([r['d3'] for r in results])

            f.write(f"Mean abs_rel:  {mean_abs_rel:.6f}\n")
            f.write(f"Mean sq_rel:   {mean_sq_rel:.6f}\n")
            f.write(f"Mean rmse:     {mean_rmse:.6f}\n")
            f.write(f"Mean rmse_log: {mean_rmse_log:.6f}\n")
            f.write(f"Mean silog:    {mean_silog:.6f}\n")
            f.write(f"Mean d1:       {mean_d1:.6f}\n")
            f.write(f"Mean d2:       {mean_d2:.6f}\n")
            f.write(f"Mean d3:       {mean_d3:.6f}\n")

    print(f"\nSaved evaluation results to:\n{output_txt}")
