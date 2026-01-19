import cv2

import open3d as o3d
import numpy as np

###############
## Utilities ##
###############

def mirror_pc(point_cloud):
    """
    Mirrors a given point cloud using
    a numpy array transformation.
    Is not currently used in the pipeline.
    """
    flip_transform = np.array([
        [-1,  0,  0,  0],
        [ 0,  1,  0,  0],
        [ 0,  0,  1,  0],
        [ 0,  0,  0,  1]
    ])

    point_cloud.transform(flip_transform)

    return point_cloud

##################
## Segmentation ##
##################

def segment_object_from_depth(depth):
    """
    Reads depth map containing coin/seal-like object
    and isolates it from its background plane.
    """

    # Creates copy of depth map
    d = depth.copy()

    if not np.isfinite(d).all():
        d[~np.isfinite(d)] = np.nanmedian(d)

    # Normalizes for thresholding
    d_norm = (d - d.min()) / (d.max() - d.min() + 1e-8)
    img = (d_norm * 255).astype("uint8")

    # Applies Otsu Thresholding
    _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    h, w = binary.shape

    # Calculates object center
    cy0, cy1 = h // 4, 3 * h // 4
    cx0, cx1 = w // 4, 3 * w // 4
    center = binary[cy0:cy1, cx0:cx1]

    # Initiates and calculates black and white components
    # i.e. object / background information
    white = np.count_nonzero(center == 255)
    black = np.count_nonzero(center == 0)

    if white >= black:
        seal_mask = (binary == 255)
    else: 
        seal_mask = (binary == 0)

    kernel = np.ones((7, 7), np.uint8)

    # Cleans produced mask by eroding 
    # then dilating using given kernel
    mask_clean = cv2.morphologyEx(seal_mask.astype("uint8"), cv2.MORPH_OPEN, kernel)
    
    # Computes connected components
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_clean, connectivity=8)

    if num_labels > 1:
        areas = stats[1:, cv2.CC_STAT_AREA]
        largest = 1 + np.argmax(areas)
        mask_final = (labels == largest)
    else:
        mask_final = mask_clean.astype(bool)

    # Further dilates mask
    mask_final = cv2.dilate(mask_final.astype("uint8"),np.ones((3,3)), iterations=1).astype(bool)
    
    return mask_final.astype(bool)

############################
## Point Cloud Generation ##
############################

def depth_to_pcd(depth, mask, rgb_image, pixel_size=1.0, relief_scale=8.0):
    """
    Generates a 3D point cloud from a 
    provided depth map
    
    :param depth: Provided depth map
    :param mask: Corresponding mask
    :param rgb_image: Corresponding rgb image
    :param pixel_size: XY spacing in world units
    :param relief_scale: Degree of relief exaggeration
    """
    # Creates a copy of depth map
    # and crops it based on given mask
    depth_masked = depth.copy()
    depth_masked[~mask] = np.nan

    # Computes object relief
    # (depth relative to its min)
    base = np.nanmin(depth_masked)
    depth_relief = depth_masked - base

    # Extracts valid 3D points
    h, w = depth.shape
    valid = mask & np.isfinite(depth_relief)
    ys, xs = np.where(valid)
    zs = depth_relief[valid] * relief_scale

    # Converts pixel coordinates to world coordinates
    X = (xs - w / 2) * pixel_size
    Y = (h / 2 - ys) * pixel_size
    pts = np.column_stack((X,Y,zs))

    # Extracts point cloud colors from rgb image
    colors = rgb_image[ys,xs] / 255.0

    # Computes normals from depth gradients
    dy, dx = np.gradient(depth_relief)
    dzdx = dx[valid]
    dzdy = dy[valid]

    normals = np.stack([-dzdx, -dzdy, np.ones_like(dzdx)], axis=1)
    norm_len = np.linalg.norm(normals, axis=1, keepdims=True) + 1e-8
    normals = normals / norm_len

    # Generates point cloud
    pcd = o3d.geometry.PointCloud()

    # Computes points
    pcd.points = o3d.utility.Vector3dVector(pts)

    # Applies colors to points
    pcd.colors = o3d.utility.Vector3dVector(colors)

    # Computes normals
    pcd.normals = o3d.utility.Vector3dVector(normals)

    # Orients normals
    pcd.orient_normals_towards_camera_location([0.0, 0.0, -1.0])

    return pcd, pts, colors