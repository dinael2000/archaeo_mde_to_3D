import os
import pymeshfix

import open3d as o3d
import numpy as np
import pyvista as pv

from utils_colors import *

###############
## Utilities ##
###############

def o3d_to_pv(mesh):
    """
    Turns provided Open3D Triangle mesh to PyVista PolyData mesh
    """
    vertices = np.asarray(mesh.vertices)
    faces = np.hstack([np.full((len(mesh.triangles), 1), 3), np.asarray(mesh.triangles)])

    pv_mesh = pv.PolyData(vertices, faces)
    if mesh.has_vertex_colors():
        pv_mesh["RGB"] = np.asarray(mesh.vertex_colors)
    
    return pv_mesh

def pv_to_o3d(mesh):
    """
    Turns provided PyVista PolyData mesh to Open3D Triangle mesh
    """
    vertices = np.asarray(mesh.points)
    faces = np.asarray(mesh.faces).reshape(-1, 4)[:, 1:]

    o3d_mesh = o3d.geometry.TriangleMesh()
    o3d_mesh.vertices = o3d.utility.Vector3dVector(vertices)
    o3d_mesh.triangles = o3d.utility.Vector3iVector(faces)
    
    # Ensures transfer of colors
    if "RGB" in mesh.array_names:
        o3d_mesh.vertex_colors = o3d.utility.Vector3dVector(np.asarray(mesh["RGB"]))
    
    # Cleans mesh and orients normals
    o3d_mesh.remove_unreferenced_vertices()
    o3d_mesh.orient_triangles()
    o3d_mesh.compute_vertex_normals()

    return o3d_mesh

def mean_edge_length(mesh):
    """
    Calculates the average length of all edges
    in a triangular mesh. Returnes average 
    of combined edge lengths into single array
    """
    V = np.asarray(mesh.vertices)
    T = np.asarray(mesh.triangles)

    e0 = np.linalg.norm(V[T[:, 0]] - V[T[:, 1]], axis=1)
    e1 = np.linalg.norm(V[T[:, 1]] - V[T[:, 2]], axis=1)
    e2 = np.linalg.norm(V[T[:, 2]] - V[T[:, 0]], axis=1)

    return float(np.mean(np.concatenate([e0, e1, e2])))


##############
## Pipeline ##
##############

def process_object_merge(mesh_obv, mesh_rev, output_path, output_path_scaled, gap_factor=0.5, scale=True, scale_factor=0.25):
    """
    Merges obverse and reverse mesh of object

    :param gap_factor: Multiplier applied to the mean
    triangle edge length to compute the Z-gap between
    the obverse and reverse meshes. An arbitrary seperation
    that does not correspond to real-world object thickness.
    0.25 -> Tight seperation
    0.5 -> Median value
    1.0 -> One full triangle edge
    2.0 -> Very conservative gap

    :param scale_factor: Factor by which to scale down (or up)
    a given object, to better represent physical dimensions.
    Recommended value <0.25
    """
    # Cleans meshes
    for mesh in (mesh_obv, mesh_rev):
        mesh.remove_duplicated_vertices()
        mesh.remove_degenerate_triangles()
        mesh.remove_duplicated_triangles()
        mesh.remove_non_manifold_edges()
        mesh.remove_unreferenced_vertices()
        mesh.compute_vertex_normals()

    # Examines whether meshes have vertex colors 
    ensure_vertex_colors(mesh_obv)
    ensure_vertex_colors(mesh_rev)

    # Flips mesh_rev
    R = mesh_rev.get_rotation_matrix_from_xyz((np.pi, 0.0, np.pi))
    mesh_rev.rotate(R, center=mesh_rev.get_center())

    # Aligns meshes
    obv_c = mesh_obv.get_center()
    rev_c = mesh_rev.get_center()
    mesh_rev.translate((obv_c[0] - rev_c[0], obv_c[1] - rev_c[1], 0.0))

    # Stacks in Z with gap
    obv_bbox = mesh_obv.get_axis_aligned_bounding_box()
    rev_bbox = mesh_rev.get_axis_aligned_bounding_box()

    # Creates gap based on relief height
    obv_height = obv_bbox.get_max_bound()[2] - obv_bbox.get_min_bound()[2]
    rev_height = rev_bbox.get_max_bound()[2] - rev_bbox.get_min_bound()[2]
    avg_height = 0.5 * (obv_height + rev_height)

    # Fallback in pathological cases (flat mesh)
    if avg_height <= 0:
        avg_height = mean_edge_length(mesh_obv)

    gap = gap_factor * avg_height

    print(
        f"[{os.path.basename(output_path)}] "
        f"obv_h={obv_height:.6g}, rev_h={rev_height:.6g}, gap={gap:.6g}"
    )

    z_shift = obv_bbox.min_bound[2] - rev_bbox.max_bound[2] - gap
    mesh_rev.translate((0, 0, z_shift))

    # Merges the meshes
    combined_mesh = mesh_obv + mesh_rev

    # Cleans merged mesh and calculates normals
    combined_mesh.remove_duplicated_vertices()
    combined_mesh.remove_unreferenced_vertices()
    combined_mesh.orient_triangles()
    combined_mesh.compute_vertex_normals()

    # Caches colors
    if not combined_mesh.has_vertex_colors():
        raise RuntimeError("Combined mesh has no vertex colors")
    src_vertices = np.asarray(combined_mesh.vertices)
    src_colors = np.asarray(combined_mesh.vertex_colors)

    # Converts to PyVista mesh
    pv_mesh = o3d_to_pv(combined_mesh)
    pv_mesh = pv_mesh.clean()

    # Applies MeshFix
    mf = pymeshfix.MeshFix(pv_mesh)
    mf.repair(verbose=True, joincomp=True, remove_smallest_components=False)
    repaired_mesh = mf.mesh

    # Converts back to Open3D
    final_mesh = pv_to_o3d(repaired_mesh)

    # Reprojects Colors
    reproject_vertex_colors(src_vertices, src_colors, final_mesh)

    # Saves mesh
    o3d.io.write_triangle_mesh(output_path, final_mesh)

    # Scales mesh
    if scale:
    
        final_mesh.scale(scale_factor, center=final_mesh.get_center())
        final_mesh.orient_triangles()
        final_mesh.compute_vertex_normals()

        o3d.io.write_triangle_mesh(output_path_scaled, final_mesh)

def batch_process_merge(input_dir, output_dir, gap_factor=0.5, scale=True, scale_factor=0.25):
    """
    Batch processes meshes
    """
    # Initiates directories
    os.makedirs(output_dir, exist_ok=True)

    files = os.listdir(input_dir)
    obv_files = [f for f in files if f.endswith("-obv.obj")]

    for obv_file in obv_files:
        stem = obv_file.replace("-obv.obj", "")
        rev_file = f"{stem}-rev.obj"

        obv_path = os.path.join(input_dir, obv_file)
        rev_path = os.path.join(input_dir, rev_file)

        if not os.path.exists(rev_path):
            print(f"Missing reverse mesh for {stem}. Skipping object.")
            continue

        print(f"[INFO] Processing {stem}")

        mesh_obv = o3d.io.read_triangle_mesh(obv_path)
        mesh_rev = o3d.io.read_triangle_mesh(rev_path)

        output_path_merged = os.path.join(
            output_dir, f"{stem}-merged.obj"
        )

        output_path_scaled = os.path.join(
            output_dir, f"{stem}-merged_scaled.obj"
        )
        
        process_object_merge(
            mesh_obv, mesh_rev, output_path_merged, output_path_scaled,
            gap_factor=gap_factor, scale=scale, scale_factor=scale_factor
        )

if __name__ == "__main__":
    input_dir = r"3_3D_Model_Creation/3d_models_open_back_TEST_19122025"
    output_dir = r"3_3D_Model_Creation/3d_models_merged_19122025"

    batch_process_merge(input_dir, output_dir, gap_factor=0.0, scale=True, scale_factor=0.25)