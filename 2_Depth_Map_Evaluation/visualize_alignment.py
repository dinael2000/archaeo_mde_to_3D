# <------------------------------------------------------
# A module with functions that permit te visualization of 
# the produced depth maps before and after alignment with
# the gt depth maps, to function as a debugging step.
# ------------------------------------------------------>

import cv2

import numpy as np

def normalize_for_vis(depth):
    """
    A function that normalizes a given
    depth map for the purposes of visualization.
    """
    d = depth.copy().astype(np.float32)
    mask = np.isfinite(d)
    if mask.sum() == 0:
        return np.zeros_like(d, np.uint8)
    mn, mx = d[mask].min(), d[mask].max()
    if mx == mn:
        return np.zeros_like(d, np.uint8)
    d = (d - mn) / (mx - mn)
    return (d * 255).clip(0,255).astype(np.uint8)


def colorize_error_map(err):
    """
    A function that applies a color to a
    given error map, for more comprehensive
    visualization.
    """
    err = err.astype(np.float32)
    err = err / (np.percentile(err,95)+1e-6)
    err = np.clip(err,0,1)
    return cv2.applyColorMap((err*255).astype(np.uint8), cv2.COLORMAP_JET)

def to_vis(img, is_depth=True):
    """
    A function that ensures that a given
    depth map / image is normalized
    and can be used for debugging.
    """
    if is_depth:
        img = normalize_for_vis(img)
    else:
        img = img.astype(np.uint8)
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return cv2.resize(img, (W_g, H_g), interpolation=cv2.INTER_NEAREST)

def create_label(img, text):
    """
    A function that creates a canvas
    for a label to be applied to a 
    plot.
    """
    h, w = img.shape[:2]
    canvas = np.zeros((h+40, w, 3), np.uint8)
    canvas[40:] = img
    cv2.putText(canvas, text, (10,30),cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)
    return canvas

def visualize_alignment_steps(pred_raw, pred_geo, pred_fixed, pred_aligned, gt, save_path, title="Alignment Debug"):
    """
    A function that produces visualizations of
    pred to gt depth maps before and 
    after editing, to function as
    debug that ensures correct alignment between the two.
    """
    H_g, W_g = gt.shape

    # Ensure that given depth maps
    # are normalized
    raw_vis = to_vis(pred_raw)
    geo_vis = to_vis(pred_geo)
    fixed_vis = to_vis(pred_fixed)
    aligned_vis = to_vis(pred_aligned)
    gt_vis = to_vis(gt)

    # Colorize depth maps 
    # for visualization
    raw_err = to_vis(colorize_error_map(
                         np.abs(cv2.resize(pred_raw.astype(np.float32),(W_g,H_g)) - gt)),
                         is_depth=False)
    geo_errv = to_vis(colorize_error_map(
                         np.abs(cv2.resize(pred_geo.astype(np.float32),(W_g,H_g)) - gt)),
                         is_depth=False)
    fixed_err = to_vis(colorize_error_map(
                         np.abs(cv2.resize(pred_fixed.astype(np.float32),(W_g,H_g)) - gt)),
                         is_depth=False)
    aligned_err = to_vis(colorize_error_map(
                         np.abs(cv2.resize(pred_aligned.astype(np.float32),(W_g,H_g)) - gt)),
                         is_depth=False)

    zero_err = to_vis(colorize_error_map(np.zeros((H_g,W_g), np.float32)),
                         is_depth=False)

    # Create debug plot
    # with appropriate labeling
    row1 = np.hstack([
        create_label(raw_vis, "1. Raw Pred"),
        create_label(geo_vis, "2. Resized (Geo)"),
        create_label(fixed_vis, "3. Flipped"),
        create_label(aligned_vis, "4. Scale+Shift"),
        create_label(gt_vis, "5. GT")
    ])

    row2 = np.hstack([
        create_label(raw_err, "Raw Err"),
        create_label(geo_err, "Geo Err"),
        create_label(fixed_err, "Flip Err"),
        create_label(aligned_err, "Final Err"),
        create_label(zero_err, "")
    ])

    final = np.vstack([row1, row2])

    pad_top = np.zeros((70, final.shape[1], 3), np.uint8)
    cv2.putText(pad_top, title, (20,45), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255,255,255), 3)
    
    out = np.vstack([pad_top, final])

    # Save debug plot
    cv2.imwrite(str(save_path), out)

    print("[debug] Saved alignment visualization: ", save_path)
