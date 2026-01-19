import os
import cv2
import pymeshfix
import trimesh

import numpy as np
import open3d as o3d

from utils_colors import *
from utils_pointcloud import *

#####################
## Mesh Generation ##
#####################

def poisson_mesh_from_pointcloud_watertight(pcd, depth):
    """
    Generates a mesh from a provided point cloud
    using Poisson reconstruction for a watertight surface
    """

    # Generates mesh
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd,depth=depth,linear_fit=False, scale=1.5)

    # Removes low-density vertices
    # for noise-trimming
    densities = np.asarray(densities)
    
    cutoff = np.mean(densities) - 0.5 * np.std(densities)
    keep = np.where(densities > cutoff)[0]
    mesh = mesh.select_by_index(keep)

    # Cleans mesh
    mesh.remove_duplicated_vertices()
    mesh.remove_unreferenced_vertices()
    mesh.remove_degenerate_triangles()

    return mesh

#####################
## Mesh Correction ##
#####################

def fix_mesh(mesh):
    """
    A function that corrects major errors
    with the produced mesh to create
    an object that will be as watertight 
    as possible
    """
    # Transform Open3D mesh to Trimesh mesh
    vertices_o3d = np.asarray(mesh.vertices)
    faces_o3d = np.asarray(mesh.triangles)
    tm = trimesh.Trimesh(vertices_o3d, faces_o3d, process=False)

    # Remove floating geometry
    components = tm.split(only_watertight=False)
    # tm = max(tm.split(only_watertight=False), key=lambda m: m.volume)
    if len(components) > 1:
        tm = max(components, key=lambda m: m.faces.shape[0])

    # Repair using Pymesh
    mf = pymeshfix.MeshFix(tm.vertices, tm.faces)
    mf.repair(verbose=False, joincomp=True)

    # Return to Open3D
    mesh = o3d.geometry.TriangleMesh(
        vertices=o3d.utility.Vector3dVector(mf.v),
        triangles=o3d.utility.Vector3iVector(mf.f)
    )

    return mesh


##############
## Pipeline ##
##############

def process_object_watertight(depth_path, rgb_path, pixel_size=1.0, relief_scale=7.0, poisson_depth=9):
    """
    Generates a colored mesh from a provided depth map
    """
    # Loads depth map and corresponding rgb image
    depth = np.load(depth_path).astype(np.float32)
    rgb_bgr = cv2.imread(rgb_path, cv2.IMREAD_COLOR)

    if rgb_bgr is None:
        raise RuntimeError(f"Could not load RGB image: {rgb_path}")
    
    # Inverts to RGB color scheme
    rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)

    # Masks object
    mask = segment_object_from_depth(depth)

    # Generates point cloud
    pcd, pts, colors = depth_to_pcd(depth, mask, rgb, pixel_size=pixel_size, relief_scale=relief_scale)

    # Generates mesh
    mesh = poisson_mesh_from_pointcloud_watertight(pcd, depth=poisson_depth)

    tris = np.asarray(mesh.triangles)

    mesh.triangles = o3d.utility.Vector3iVector(tris[:, [0, 2, 1]])

    # Fixes geometry
    mesh = fix_mesh(mesh)

    # Colors mesh
    mesh = transfer_vertex_colors_fast(mesh, pts, colors)

    return mesh

def batch_process_watertight(depth_dir, rgb_dir, output_dir, output_format="obj", pixel_size=1.0, relief_scale=7.0, poisson_depth=9):
    """
    Batch process dataset of seals/coins to produce 3D meshes
    
    :param depth_dir: Directory to depth maps
    :param rgb_dir: Directory to rgb images
    :param output_dir: Directory to save results
    :param output_format: Format to save 3D models (e.g. obj or ply)
    """
    # Initiates directories
    os.makedirs(output_dir, exist_ok=True)

    # Reads depth files
    depth_files = [f for f in os.listdir(depth_dir) if f.endswith(".npy")]
    depth_files.sort()

    # Processes depth maps
    for df in depth_files:
        depth_path = os.path.join(depth_dir, df)

        stem = df.replace("_depth.npy", "")
        
        extensions = [os.path.join(rgb_dir, stem+".jpg"), os.path.join(rgb_dir,stem+".png")]
        rgb_path = next((p for p in extensions if os.path.exists(p)), None)
        
        if rgb_path is None:
            print(f"Failed to read rgb image for {stem}")
            continue


        out_name = f"{stem}.{output_format}"
        out_path = os.path.join(output_dir, out_name)

        print(f"Processing {stem}")

        try: 
            mesh = process_object_watertight(depth_path, rgb_path, pixel_size=pixel_size, relief_scale=relief_scale, poisson_depth=poisson_depth)
            o3d.io.write_triangle_mesh(out_path, mesh)
            print(f"Saved {out_path}")
        except Exception as e:
            print(f"Failed to process {df}: {e}")

if __name__ == "__main__":
    depth_dir = r"1_DA1_Depth_Estimation/result_depth_maps/depth_npy"
    rgb_dir = r"data_4_17"
    out_dir = r"3_3D_Model_Creation/3d_models_watertight"

    batch_process_watertight(depth_dir, rgb_dir, out_dir, "obj", pixel_size=1.0, relief_scale=7.0, poisson_depth=9)