# < -------------------------------------------------------------------------
# A module with functions to calculate detail metrics 
# for the evaluation of monocular depth estimation depth maps 
# (affive invariant) to ground truth depth maps (metric).
# 
# Citations: See citations.txt
# ------------------------------------------------------------------------- >

import cv2

import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from masking import *
from alignment import *

#########################
## Auxiliary Functions ##
#########################

def compute_gradients(depth):
    """
    Computes spatial gradients of input depth map
    along the x- and y-directions, i.e. how quickly 
    pixel values change horizontally and vertically,
    using a Sobel filter for edge-detection.
    
    :param depth: Depth map
    :return dx:
    :return dy: 
    """
    dx = cv2.Sobel(depth, cv2.CV_32F, 1, 0, ksize=3)
    dy = cv2.Sobel(depth, cv2.CV_32F, 0, 1, ksize=3)
    return dx, dy


def compute_laplacian(depth):
    """
    Computes the Laplacian (∇(^2)z) of a given depth map. 
    Used for shape analysis.

    Positive values -> Local concave curvature
    Negative values -> local convex curvature
    Near zero -> Flat of smoothly varying regions
    
    :param depth: Depth map to be analyzed

    :return: A float32 2D array with local curvature values
    """
    return cv2.Laplacian(depth, cv2.CV_32F, ksize=3)


def compute_normals(depth):
    """
    Converts a depth map to a per-pixel surface
    normal map by computing depth gradients.
    
    :param depth: Depth map
    :return: Normalized vector as 3-channel image
    """
    dx, dy = compute_gradients(depth)
    nx = -dx
    ny = -dy
    nz = np.ones_like(depth)

    norm = np.sqrt(nx**2 + ny**2 + nz**2) + 1e-8

    return np.stack((nx/norm, ny/norm, nz/norm), axis=-1)


def normal_angle_error(n_pred, n_gt):
    """
    Computes per-pixel angular error between predicted 
    and ground truth depth maps using the dot product.
    
    :param n_pred: Predicted depth map normals
    :param n_gt: Ground truth depth map normals
    :return: Angular error in radians
    """
    dot = np.sum(n_pred * n_gt, axis=-1)

    # Clips numerical errors to ensure safe
    # angle computation
    dot = np.clip(dot, -1.0, 1.0)

    return np.arccos(dot)


def save_heatmap(image, title, save_path, vmax=None):
    """
    Creates and saves colored heatmap
    
    :param image: Input image
    :param title: Title of heatmap
    :param save_path: Path for saving the heatmap
    :param vmax: Upper bound of the color scale
    
    """

    plt.figure(figsize=(5, 5))

    # Lower bound of the color scale
    vmin = 0.0

    # Upper bound of the color scale
    # avoiding invalid input
    # to the 99th percentile
    if vmax is None:
        vmax = np.nanpercentile(image, 99)

    # Visualizes the heatmap
    plt.imshow(image, cmap="turbo", vmin=vmin, vmax=vmax)
    plt.title(title)
    plt.axis("off")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    
####################
## Detail Metrics ##
####################

def compute_detail_metrics(gt, pred, mask, pred_name, out_dir):
    """
    Computes detail error metrics for predicted depth map 
    (affine invariant) against ground truth depth map
    (metric) within a given relief mask.

    1. Gradient Magnitude Error (first-order detail): 
    Measures how well local slopes match.

    2. Laplacian Error (second-order detail):
    Measures shape sharpness fidelity along curvature.

    3. Normal Error:
    Measures per-pixel angle between pred and gt surface normals
    in radians.

    -> Consider adding SSIM?

    :param gt: Ground truth depth map
    :param pred: Predicted depth map
    :param mask: Relief map
    :param pred_name: Object name
    :param out_dir: Path for saving metrics
    
    """
    # Initializes output directory
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Resizes pred to shape of gt
    if pred.shape != gt.shape:
        pred = cv2.resize(pred, (gt.shape[1], gt.shape[0]))

    # Builds masks within gt
    mask_valid = gt > 0
    relief_valid = mask_valid & mask

    # Computers Gradient Magnitude Error
    gx_gt, gy_gt = compute_gradients(gt)
    gx_pr, gy_pr = compute_gradients(pred)
    grad_err = np.sqrt((gx_gt-gx_pr)**2 + (gy_gt-gy_pr)**2)

    # Computes Laplacian Error
    lap_gt = compute_laplacian(gt)
    lap_pr = compute_laplacian(pred)
    lap_err = np.abs(lap_gt - lap_pr)

    # Computes Normal Error
    n_gt = compute_normals(gt)
    n_pr = compute_normals(pred)
    normal_err = normal_angle_error(n_pr, n_gt)

    # Visualizations

    # Gradient Error Heatmap
    grad_err_vis = np.full_like(grad_err, np.nan, dtype=np.float32)
    grad_err_vis[relief_valid] = grad_err[relief_valid]
    save_heatmap(
        grad_err_vis,
        "Gradient Error",
        out / f"{pred_name}_grad.png"
    )

    # Laplacian Error Heatmap
    lap_err_vis = np.full_like(lap_err, np.nan, dtype=np.float32)
    lap_err_vis[relief_valid] = lap_err[relief_valid]
    save_heatmap(
        lap_err_vis,
        "Laplacian Error",
        out / f"{pred_name}_lap.png"
    )

    # Normal Error Heatmap
    normal_err_vis = np.full_like(normal_err, np.nan, dtype=np.float32)
    normal_err_vis[relief_valid] = normal_err[relief_valid]
    save_heatmap(
        normal_err_vis,
        "Normal Error",
        out / f"{pred_name}_normal.png",
        vmax=0.5
    )

    # Computes region masks
    m_center, m_mid, m_boundary = build_region_masks(gt)

    # Restricts region masks to surface relief
    m_center &= relief_valid
    m_mid &= relief_valid
    m_boundary &= relief_valid

    # Computes per region metrics
    def metric(err, m):
        v = err[m]
        return np.mean(v) if len(v)>0 else np.nan

    results = {
        "Grad_center": metric(grad_err, m_center),
        "Grad_mid":    metric(grad_err, m_mid),
        "Grad_bound":  metric(grad_err, m_boundary),

        "Lap_center": metric(lap_err, m_center),
        "Lap_mid":    metric(lap_err, m_mid),
        "Lap_bound":  metric(lap_err, m_boundary),

        "Normal_center": metric(normal_err, m_center),
        "Normal_mid":    metric(normal_err, m_mid),
        "Normal_bound":  metric(normal_err, m_boundary),
    }

    # Saves results in a LaTeX table
    with open(out/f"{pred_name}_detail.tex", "w", encoding="utf-8") as f:
        f.write("\\begin{table}[h]\n\\centering\n")
        f.write("\\caption{Detail metrics for "+pred_name+"}\n")
        f.write("\\begin{tabular}{lccc}\n")
        f.write("Metric & Center ↓ & Mid ↓ & Boundary ↓\\\\\n\\midrule\n")
        f.write(f"Grad RMSE & {results['Grad_center']:.4f} & {results['Grad_mid']:.4f} & {results['Grad_bound']:.4f}\\\\\n")
        f.write(f"Laplacian & {results['Lap_center']:.4f} & {results['Lap_mid']:.4f} & {results['Lap_bound']:.4f}\\\\\n")
        f.write(f"Normal Err & {results['Normal_center']:.4f} & {results['Normal_mid']:.4f} & {results['Normal_bound']:.4f}\\\\\n")
        f.write("\\end{tabular}\n\\end{table}")