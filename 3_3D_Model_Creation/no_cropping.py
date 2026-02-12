import cv2
import os

import numpy as np
import open3d as o3d

import matplotlib.pyplot as plt

def depth_to_pointcloud_orthographic(depth_map, color_image, mask_image=None, scale_factor=255):
    """
        A function that takes a depth map and color image as input
        and produces a 3D point cloud using an orthographic projection.
        The orthographic projection treats the depth values as direct
        heigh values rather than distances from the camera. The x and y
        coordinates are derived directly from the pixel coordinates.    
    """

    color_image = cv2.cvtColor(color_image,cv2.COLOR_BGR2RGB)

    height, width = depth_map.shape

    # Create a grid of pixel coordinates
    y,x = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')

    # Scale the depth values
    z = (depth_map/scale_factor) * height/2

    # Create 3D points (x and y are pixel coordinates, z is from the depth map)
    points = np.stack((x,y,z), axis=-1).reshape(-1,3)

    if mask_image is not None:
        # Resize mask to depth map size
        if mask_image.shape[:2] != depth_map.shape:
            mask_resized = cv2.resize(
                mask_image.astype(np.uint8),
                (width, height),
                interpolation=cv2.INTER_NEAREST
            )
        else:
            mask_resized = mask_image.astype(np.uint8)

        # Flatten to match point order
        mask_flat = mask_resized.flatten() > 0

        # Combine mask and valid depth
        valid_depth_mask = (points[:, 2] != 0) & mask_flat

    else:
        # Default: only filter invalid (zero) depths
        valid_depth_mask = points[:, 2] != 0


    points = points[valid_depth_mask]


    # Create O3D PointCloud object
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    # o3d.visualization.draw_geometries([pcd])

    if color_image.shape[:2] != depth_map.shape:
        color_image = cv2.resize(color_image, (depth_map.shape[1], depth_map.shape[0]), interpolation=cv2.INTER_LINEAR)

    # Add colors to the point cloud
    colors = color_image.reshape(-1, 3)[valid_depth_mask] / 255.0
    pcd.colors = o3d.utility.Vector3dVector(colors)

    _, ind = pcd.remove_statistical_outlier(nb_neighbors=15, std_ratio=1)
    inlier_cloud = pcd.select_by_index(ind)

    return inlier_cloud, z, height, width

# Mirroring the point cloud on the vertical axis 

def mirror_pc(point_cloud):
    """
    A function that mirrors a 
    point cloud in the 
    """
    flip_transform = np.array([
        [-1,  0,  0,  0],
        [ 0,  1,  0,  0],
        [ 0,  0,  1,  0],
        [ 0,  0,  0,  1]
    ])

    point_cloud.transform(flip_transform)

    return point_cloud

def save_pc(point_cloud, i, save_path):
        o3d.io.write_point_cloud(f'{save_path}', point_cloud, write_ascii=True)

# Generating a 3D mesh from the point cloud with a Poisson surface reconstruction algorithm

def pointcloud_to_mesh(point_cloud):
    """
    A function that turns a point cloud 
    into an open3D TriangleMesh
    """

    point_cloud.estimate_normals()
    point_cloud.orient_normals_to_align_with_direction()

    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(point_cloud, depth=9)

    return mesh, densities

# Cleaning of the generated mesh

def cleanup_mesh(mesh):
    """
    A function that removes incorrect
    geometry and unecessary noise
    from a given mesh
    """
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_non_manifold_edges()
    # o3d.visualization.draw_geometries([mesh])
    return mesh

# Saving the mesh in local directory

def save_mesh(mesh, i, save_path):
    o3d.io.write_triangle_mesh(f'{save_path}', mesh, write_triangle_uvs = True)
    print(f"Saved {i} in {save_path}")

def batch_process_no_cropping(depth_dir, rgb_dir, output_dir, mirrored=True, output_format="obj"):
    """
    Batch process dataset of seals/coins to produce 3D meshes
    """

    # Initiates directories
    os.makedirs(output_dir, exist_ok=True)

    # Reads depth files
    depth_files = [f for f in os.listdir(depth_dir) if f.endswith(".npy")]
    depth_files.sort()

    # Processs depth maps
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
            depth_map = np.load(depth_path)
            color_image = cv2.imread(rgb_path)

            pointcloud, _, _, _ = depth_to_pointcloud_orthographic(depth_map=depth_map, color_image=color_image)
            if mirrored:
                pointcloud = mirror_pc(pointcloud)

            mesh, densities = pointcloud_to_mesh(point_cloud=pointcloud)

            cleaned_mesh = cleanup_mesh(mesh)

            save_mesh(cleaned_mesh, stem, out_path)
            
        except Exception as e:
            print(f"Failed to process {df}: {e}")

            