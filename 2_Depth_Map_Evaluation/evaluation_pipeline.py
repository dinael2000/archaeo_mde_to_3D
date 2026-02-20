# < -------------------------------------------------------------------
# Module with functions to run the evaluation pipeline
# ------------------------------------------------------------------->

import cv2
import torch

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn.functional as F

from math import exp
from pathlib import Path
from datetime import datetime

from alignment import *
from evaluation_metrics import *
from diagnostics import *
from detail_metrics import *
from visualize_alignment import *
from masking import *

##########################
## Auxilliary Functions ##
##########################

def load_depth_map(path, clip_start=1e-6, clip_end=10.0):
    """
    A function that loads and correctly reads
    a ground truth depth map (normalized) 
    scaled to a given metric range (clip_start - clip_end)
    
    :return gt_depth:
    """
    # Reads the gt depth map
    # and handles possible errors
    gt_depth = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)

    if gt_depth is None:
        raise ValueError(f"Could not load depth map: {path}")

    if gt_depth.ndim == 3:
        gt_depth = gt_depth[..., 0]

    if gt_depth.dtype == np.uint16 or gt_depth.dtype == np.float32:
        max_val = 65535.0
        gt_depth = clip_start + (gt_depth.astype(np.float32) / max_val) * (clip_end - clip_start)
    else:
        max_val = np.iinfo(gt_depth.dtype).max
        gt_depth = clip_start + (gt_depth.astype(np.float32) / max_val) * (clip_end - clip_start)

    gt_depth[np.isnan(gt_depth)] = 0
    gt_depth[np.isinf(gt_depth)] = 0

    return gt_depth

def visualize_depth(rgb_image, gt_depth, pred_colored, save_path=None):
    """
    A function that visualizes the results of the depth prediction evaluation as plots.
    
    :param rgb_image: The original RGB input image
    :param gt_depth: The Ground Truth depth map
    :param pred_depth_aligned: The Predicted depth map
    :param save_path: A path to save the produced plots
    """
    
    # Sets up the plot
    plt.figure(figsize=(12,4))
    plt.subplot(1,3,1)
    plt.imshow(cv2.cvtColor(rgb_image,cv2.COLOR_BGR2RGB))
    plt.title("RGB image")
    plt.axis('off')

    plt.subplot(1,3,2)
    masked_gt = np.ma.masked_equal(gt_depth, 0)
    im = plt.imshow(masked_gt, cmap='inferno')
    im.cmap.set_bad(color='purple')
    plt.title("GT Depth")
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.subplot(1, 3, 3)
    plt.imshow(cv2.cvtColor(pred_colored, cv2.COLOR_BGR2RGB))
    plt.title("Predicted Depth")
    plt.axis("off")
    
    # Saves the plot
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        plt.close()
    else:
        plt.show()


#########################
## End-to-end pipeline ##
#########################

def evaluate_dataset(
    rgb_dir,
    gt_dir,
    pred_dir,
    model_name,
    pred_vis_dir=None,
    visualize=False,
    save_vis_dir=None,
    save_pred_dir=None,
    save_diagnostics_dir=None,
    save_diagnostics_dir_post=None,
    debug_dir = None
):
    """
    Main function to calculate, visualize and save 
    Monocular Depth Estimation predicted depth (affine invariant)
    evaluation metrics, against ground truth depth map (metric)
    
    :param rgb_dir: Path to dir with rgb images
    :param gt_dir: Path to dir with gt depth maps
    :param pred_dir: Path to dir with pred depth maps

    :param model_name: Name of used model

    :param pred_vis_dir: Path to directory where vis. will be saved
    :param visualize: Whether visalizations will be produced (bool)

    :param save_vis_dir: Path to directory where vis. will be saved
    :param save_pred_dir: Path where analysis results will be saved

    :param save_diagnostics_dir: Path where diagnostics debug will be saved
    :param save_diagnostics_dir_post: Path where diagnostics debug will be saved
    (post alignment)

    :param debug_dir: Path where debug will be saved
    """
    # Initializes current date/time
    current_date = datetime.now().strftime("%d%m%Y")

    # Sets up input directories
    rgb_dir  = Path(rgb_dir)
    gt_dir   = Path(gt_dir)
    pred_dir = Path(pred_dir)
    debug_dir = Path(debug_dir)

    # Sets up output directories
    if save_pred_dir is not None:
        save_pred_dir = Path(save_pred_dir)
        save_pred_dir.mkdir(parents=True, exist_ok=True)

        normal_dir = save_pred_dir / "normal_maps"
        normal_dir.mkdir(parents=True, exist_ok=True)
    else:
        normal_dir = None

    if save_diagnostics_dir is not None:
        diagnostics_dir = Path(save_diagnostics_dir)
        diagnostics_dir.mkdir(parents=True, exist_ok=True)

    if save_diagnostics_dir_post is not None:
        diagnostics_dir_post = Path(save_diagnostics_dir_post)
        diagnostics_dir_post.mkdir(parents=True, exist_ok=True)

    if visualize and save_vis_dir:
        save_vis_dir = Path(save_vis_dir)
        save_vis_dir.mkdir(parents=True, exist_ok=True)

    # Initializes results list and depth predictions dictionary
    results = []
    pred_depth_dict = {}

    # <------------------------------------------------------------- Main Loop

    # Reads rgb dir
    for rgb_path in sorted(rgb_dir.iterdir()):
        stem = rgb_path.stem
        print(f"Evaluating: {stem}")

        # Loads rgb images and gt depth maps
        rgb = cv2.imread(str(rgb_path))
        gt  = load_depth_map(gt_dir / f"{stem}.tif").astype(np.float32)

        # Calculates shape of gt depth map
        H_g, W_g = gt.shape

        # Loads predicted depth map
        pred_raw = np.load(pred_dir / f"{stem}_depth.npy").astype(np.float32)

        H_g, W_g = gt.shape

        print("  GT shape  :", gt.shape)
        print("  Pred shape:", pred_raw.shape)

        if rgb is not None:
            print("  RGB shape :", rgb.shape)

        pred_pre = preprocess_pred_depth(pred_raw, gt.shape)
        pred_geo, pr_circle, gt_circle, scale_xy, M = spatial_align_radius(pred_pre, gt)

        # CHANGE Removed spatial alignment with radius
        # Initiates geometry-based spatial alignment
        
        # Builds and saves necessary masks
        mask_valid  = build_valid_mask(gt)
        mask_relief = build_relief_mask(gt)

        if visualize and save_vis_dir:
            cv2.imwrite(
                str(save_vis_dir / f"{stem}_relief_mask.png"),
                (mask_relief.astype(np.uint8) * 255),
            )

        # Checks coefficient correlation
        gt_corr   = gt[mask_valid]
        pred_corr = pred_geo[mask_valid]
        if gt_corr.size > 0 and pred_corr.size > 0:
            corr_raw = float(np.corrcoef(gt_corr, pred_corr)[0, 1])
        else:
            corr_raw = 0.0

        # Checks pred depth map orientation
        pred_fixed, corr_used, flipped = flip_depth(gt, pred_geo, mask_valid)

        # Initializes and saves diagnostics
        if diagnostics_dir is not None:
            alignment_diagnostics(gt, pred_fixed, mask_valid,
                                  diagnostics_dir, image_name=stem)

        # Initializes scale and shift alignment
        pred_aligned = align_scale_shift_lowfreq(pred_fixed, gt, mask_valid)

        # Builds masks for metrics
        mask_metrics = build_valid_mask(gt, pred_aligned)

        # Initializes and saves diagnostics post-alignment
        if diagnostics_dir_post is not None:
            alignment_diagnostics(gt, pred_aligned, mask_metrics,
                                  diagnostics_dir_post, image_name=stem)
            
        # Updates pred depth dictionary
        pred_depth_dict[rgb_path.name] = (rgb, pred_aligned)

        # Initializes and saves debug
        vis_path = debug_dir / f"{stem}_alignment_debug.png"

        visualize_alignment_steps(
            pred_raw=pred_raw,
            pred_geo=pred_geo,
            pred_fixed=pred_fixed,
            pred_aligned=pred_aligned,
            gt=gt,
            save_path=vis_path,
            title=f"Alignment Debug: {stem}"
        )

        # Initializes and saves normal maps
        if normal_dir is not None:
            normal_map = compute_normals(pred_aligned)
            normal_img = ((normal_map + 1) / 2 * 255).astype(np.uint8)
            cv2.imwrite(
                str(normal_dir / f"{stem}_normal.png"),
                cv2.cvtColor(normal_img, cv2.COLOR_RGB2BGR),
            )

        # Calculates standard depth metrics
        absrel_val = abs_rel(pred_aligned, gt, mask_metrics)
        sqrel_val = sq_rel(pred_aligned, gt, mask_metrics)
        rmse_val = rmse(pred_aligned, gt, mask_metrics)
        rmse_log_val = rmse_log(pred_aligned, gt, mask_metrics)
        si_val = silog(pred_aligned, gt, mask_metrics)
        si_100_val = silog_100(pred_aligned, gt, mask_metrics)
        d1_val = delta_accuracy(pred_aligned, gt, mask_metrics, 1.25)
        d2_val = delta_accuracy(pred_aligned, gt, mask_metrics, 1.25**2)
        d3_val = delta_accuracy(pred_aligned, gt, mask_metrics, 1.25**3)

        # Calculates detail-aware scalar metrics
        grad_rmse_val = gradient_rmse(pred_aligned, gt, mask_relief)
        lap_err_val = laplacian_error(pred_aligned, gt, mask_relief)
        norm_err_val = normal_error(pred_aligned, gt, mask_relief)
        hfer_val = highfreq_ratio(pred_aligned, gt, mask_relief)

        # Calculates combined score
        combined_detail_score = (
            exp(-norm_err_val) * 0.40 +
            exp(-grad_rmse_val) * 0.25 +
            exp(-lap_err_val) * 0.20 +
            exp(-abs(hfer_val - 1.0)) * 0.15
        )

        # Appends results dictionary
        results.append({
            "image"       : rgb_path.name,
            "AbsRel"      : absrel_val,
            "SqRel"       : sqrel_val,
            "RMSE"        : rmse_val,
            "RMSElog"     : rmse_log_val,
            "SIlog"       : si_val,
            "SIlog * 100" : si_100_val,
            "d1"          : d1_val,
            "d2"          : d2_val,
            "d3"          : d3_val,
            "Grad_RMSE"   : grad_rmse_val,
            "Lap_Error"   : lap_err_val,
            "Normal_Error": norm_err_val,
            "HF_Ratio"    : hfer_val,
            "DetailScore" : combined_detail_score,
            "CorrRaw"     : corr_raw,
            "CorrUsed"    : corr_used,
            "inverted"    : bool(flipped),
        })

        # Initiates and saves visual comparison plots
        if visualize and save_vis_dir:
            pred_vis = None
            if pred_vis_dir:
                pred_vis_path = Path(pred_vis_dir) / f"{stem}_depth_colored.png"
                if pred_vis_path.exists():
                    pred_vis = cv2.imread(str(pred_vis_path))

            if pred_vis is None:
                pred_norm = (pred_aligned - pred_aligned.min()) / (
                    pred_aligned.max() - pred_aligned.min() + 1e-8
                )
                pred_vis = (plt.cm.inferno(pred_norm)[:, :, :3] * 255).astype(np.uint8)

            vis_path = save_vis_dir / f"{stem}_vis.png"
            visualize_depth(rgb, gt, pred_vis, save_path=vis_path)

            mask_img = (mask_relief.astype(np.uint8) * 255)
            cv2.imwrite(str(save_vis_dir / f"{stem}_mask.png"), mask_img)
    # End main loop ----------------------------------------------------------------------- >

    # Initializes and saves results dataframe
    # with mean metrics
    df1 = pd.DataFrame(results)
    df_mean = df1.mean(numeric_only=True).rename("Mean Metrics")
    df = pd.concat([df1, df_mean.to_frame().T], ignore_index=True)

    if save_pred_dir is not None:
        df.to_csv(
            save_pred_dir / f"Analysis_Results_{model_name}_{current_date}.csv",
            index=True
        )

    print("\nDone!")

    return df, pred_depth_dict
