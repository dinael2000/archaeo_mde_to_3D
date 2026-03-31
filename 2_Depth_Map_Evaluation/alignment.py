# < -------------------------------------------------------------------
# Module with functions to align predicted depth maps 
# from monocular depth estimation models (scale and shift invariant)
# to ground truth depth maps (metrics).
#
# Citations: see citations.txt
# ------------------------------------------------------------------->

import cv2
import torch

import numpy as np

from detail_metrics import *

#########################
## Auxiliary Functions ##
#########################

def preprocess_pred_depth(pred_raw, gt_shape):
    """
    A function that fixes the elliptical distortion of generated depth maps
    by enforcing a square aspect ratio before alignment.
    Fixes DepthAnything-V1 elliptical distortion by enforcing a square aspect ratio
    BEFORE alignment.
    """

    H_g, W_g = gt_shape

    # Crops oblong prediction
    H_p, W_p = pred_raw.shape
    s = min(H_p, W_p)

    y0 = (H_p - s) // 2
    x0 = (W_p - s) // 2
    pred_sq = pred_raw[y0:y0+s, x0:x0+s]

    # Resized to match GT coordinate system
    pred_fixed = cv2.resize(
        pred_sq.astype(np.float32),
        (H_g, H_g),
        interpolation=cv2.INTER_LINEAR
    )

    return pred_fixed

def center_square_crop(img, target_size):
    H, W = img.shape[:2]
    s = min(H, W)

    y0 = (H - s) // 2
    x0 = (W - s) // 2

    cropped = img[y0:y0+s, x0:x0+s]

    if target_size is not None:
        cropped = cv2.resize(cropped, (target_size, target_size), interpolation=cv2.INTER_LINEAR)

    return cropped

def flip_depth(gt, pred, mask, corr_threshold=0.0):
    """
    Detects and fixes direction of a predicted depth map 
    relative to a gt depth map, to facilitate comparisons.
    If correlation(gt, pred) < corr_threshold (/given threshold), 
    the prediction is considered inverted, therefore must be multiplied by -1.
    
    :param gt: Ground-truth depth map
    :param pred: Raw predicted depth map
    :param mask: Boolean mask of valid pixels
    :param corr_threshold: Description

    :return pred_fixed: Flipped prediction (if applicable)
    :return corr: Raw correlation before flipping
    :return flipped: Boolean operator to confirm whether flipped or not     
    """

    # Extracts valid pixels from depth maps
    gt_valid   = gt[mask].astype(np.float32)
    pred_valid = pred[mask].astype(np.float32)
    
    # If invalid depth maps, returns unmodified input
    if gt_valid.size == 0 or pred_valid.size == 0:
        return pred, 0.0, False

    gt_std = gt_valid.std()
    pr_std = pred_valid.std()

    if gt_std < 1e-6 or pr_std < 1e-6:
        return pred, 0.0, False

    # Computes correlation
    corr = float(np.corrcoef(gt_valid, pred_valid)[0, 1])
    
    flipped = False # Flipped boolean flag

    # Compares corr to threshold and flips if necessary
    if corr < corr_threshold:
        pred    = -pred
        flipped = True # Flipped boolean flag, updated

    return pred, corr, flipped


def smooth_field(depth_map, sigma=25):
    """
    Applies large Gaussian blur (sigma = 25) to input depth map.
    Returns low-frequency structure of the original depth map.
    
    :param depth_map: Depth map
    :param sigma: Standard deviation of the Gaussian kernel
    Here, very large for harsh bluring of details.

    :return: Low-frequency depth map
    """
    k = int(sigma * 4) | 1
    return cv2.GaussianBlur(depth_map, (k, k), sigma)

def fit_circle_from_mask(mask):
    """
    Fits a circle to given binary mask of circular rim pixels
    
    :param mask: Given mask

    :return (cx, cy): Estimated center of circle
    :return R: Estimated radius
    """
    # Extracts rim pixel coordinates
    ys, xs = np.where(mask)

    # Fallback in case of small mask (<20 rim pixels)
    if xs.size < 20:
        H, W = mask.shape
        return (W / 2.0, H / 2.0, min(H, W) / 2.0)

    # Builds linearized circle-fitting system
    A = np.column_stack([xs, ys, np.ones_like(xs)])
    b = -(xs**2 + ys**2)

    # Solves system using least squares
    c, *_ = np.linalg.lstsq(A, b, rcond=None)

    # Converts algebraic parameters to center and radius
    cx = -c[0] / 2.0
    cy = -c[1] / 2.0
    R  = np.sqrt((c[0]**2 + c[1]**2) / 4.0 - c[2])

    return cx, cy, R

###################
## Detecting Rim ## 
###################


def detect_rim_lap(depth, ksize=3, rim_percentile=90):
    """
    Detects rim of coin-like object from  depth map by computing 
    Laplacian magnitude, to indicate sharp curvature variations.
    
    :param depth: Ground truth depth map (pref. ground truth)
    :param ksize: Kernel size
    :param rim_percentile: Percentage by which to eliminate Laplacian responses.
    Weakest n% are eliminated, as corresponding to low-curvature values.
    
    :return: Boolean binary mask of pixels that belong to the rim
    """

    # Computes the Laplacian of the depth map
    lap = cv2.Laplacian(depth.astype(np.float32), cv2.CV_32F, ksize=ksize)
    lap_abs = np.abs(lap)

    # Picks strongest Laplacian responses 
    thr = np.percentile(lap_abs, rim_percentile)
    mask = lap_abs >= thr

    # Cleans mask using morphological opening
    # to remove small noise and smooth edges, 
    # for a continuous rim mask
    mask = cv2.morphologyEx(
        mask.astype(np.uint8),
        cv2.MORPH_OPEN,
        np.ones((5, 5), np.uint8)
    ).astype(bool)

    return mask

###########################
## Scale-Shift Alignment ##
###########################


def align_scale_shift_lowfreq(pred, gt, mask, sigma=25):
    """
    Aligns low-frequency shape components of ground truth 
    and predicted depth maps, in order to preserve local relief. 
    Applies linear scale and shift, to accommodate scale-shift invariant
    nature of pred depth maps, for better alignment to metric gt depth maps.
    
    :param pred: Predicted depth map
    :param gt: Ground truth depth map
    :param mask: Boolean mask of valid pixels
    :param sigma: Standard deviation of the Gaussian kernel
    
    :return: Scale and shift applied to prediction
    """
    # Calls smooth_field function to smooth depth maps
    # and preserve local detail
    pred_s = smooth_field(pred, sigma)
    gt_s   = smooth_field(gt,   sigma)

    # Extracts valid pixels from depth maps
    p = pred_s[mask].reshape(-1)
    g = gt_s[mask].reshape(-1)

    # Fallback in case of invalid input
    if p.size == 0 or g.size == 0:
        return pred

    # Computes means
    p_mean = p.mean()
    g_mean = g.mean()
    
    # Computes covariance-like numerator
    num = np.sum((p - p_mean) * (g - g_mean))

    # Computes variance-like denominator
    den = np.sum((p - p_mean) ** 2)

    # Computes scale (a) and shift (b)
    if den == 0:
        a = 1.0
        b = g_mean - a * p_mean
    else:
        a = num / den
        b = g_mean - a * p_mean

    return a * pred + b

######################
## Radius alignment ##
######################

def spatial_align_radius(pred, gt):
    """
    Alignes predicted depth map to ground truth depth map
    by matching masks of rim circles, for coin-like objects.
    
    :param pred: Predicted depth map (affine invariant)
    :param gt: Ground truth depth map (metric)
    
    :return pred_warp: Float32 array of pred depth map
    wraped into gt coordinates
    :return (cx_p, cy_p, R_p): Detected circle-structure in predicted depth map
    :return (cx_g, cy_g, R_g): Detected circle-structure in ground truth
    :return scale_xy: Scalar spatial scale
    :return M: 2x3 affine matrix
    """

    # Calls detect_rim_lap function to 
    # create rim mask of coin-like objects 
    # using Laplacian magnitude
    rim_gt = detect_rim_lap(gt)
    rim_pred = detect_rim_lap(pred)

    # Calls fit_circle_from_mask function
    # to detect circle from rim mask
    cx_g, cy_g, R_g = fit_circle_from_mask(rim_gt)
    cx_p, cy_p, R_p = fit_circle_from_mask(rim_pred)

    # Raises error in case of invalid radius 
    # for detected circle in pred depth map
    if R_p < 1e-6:
        raise RuntimeError("Invalid predicted depth map")

    # Scales and transforms predicted depth map
    # to ground truth coordinates
    scale_xy = R_g / R_p
    tx = cx_g - scale_xy * cx_p
    ty = cy_g - scale_xy * cy_p

    # Calculates affine transformation matrix
    M = np.array([
        [scale_xy, 0.0,      tx],
        [0.0,      scale_xy, ty]
    ], dtype=np.float32)

    # Calculates gt shape
    H_g, W_g = gt.shape

    # Warps pred to gt coordinates
    # based on affine transformation
    pred_warp = cv2.warpAffine(
        pred.astype(np.float32),
        M,
        (W_g, H_g),
        flags=cv2.INTER_LINEAR
    )

    return pred_warp, (cx_p, cy_p, R_p), (cx_g, cy_g, R_g), scale_xy, M
