import open3d as o3d
import numpy as np

from scipy.spatial import cKDTree

def transfer_vertex_colors(mesh, pcd_points, pcd_colors):
    """
    Transfers colors from point cloud to mesh
    by sampling nearest points between them
    """
    # Filters invalid points
    finite = np.isfinite(pcd_points).all(axis=1)
    pts = pcd_points[finite]
    colors = pcd_colors[finite]

    # Builds KD-tree for nearest-neighbor computation
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(pts)
    kd = o3d.geometry.KDTreeFlann(pc)

    mesh_colors = []

    # Finds nearest point for every mesh vertex
    for v in mesh.vertices:
        v = np.asarray(v)
        k, idx, _ = kd.search_knn_vector_3d(v,1)
        # Assings colors to points
        if k < 1:
            mesh_colors.append([0.7, 0.7, 0.7])
        else:
            mesh_colors.append(colors[idx[0]])

    # Applies vertex colors to the mesh
    mesh.vertex_colors = o3d.utility.Vector3dVector(mesh_colors)

    return mesh

def transfer_vertex_colors_fast(mesh, pts, colors):
    """
    Transfers colors from point cloud to mesh
    by sampling nearest points between them
    """
    finite = np.isfinite(pts).all(axis=1)
    pts = pts[finite]
    colors = colors[finite]

    verts = np.asarray(mesh.vertices)

    tree = cKDTree(pts)

    dists, idx = tree.query(verts, k=1)

    mesh.vertex_colors = o3d.utility.Vector3dVector(colors[idx])

    return mesh

def ensure_vertex_colors(mesh, fallback=(1.0, 1.0, 1.0)):
    """
    Ensures transfer of vertex colors between meshes.
    If one of the provided meshes does not have
    vertex colors, it uses a provided color value
    as fallback.
    """
    if not mesh.has_vertex_colors():
        n = len(mesh.vertices)
        mesh.vertex_colors = o3d.utility.Vector3dVector(np.tile(np.array(fallback), (n, 1)))

def reproject_vertex_colors(src_vertices, src_colors, dst_mesh):
    """
    Projects colors onto mesh vertices
    """
    # Examines validity of source mesh
    # vertices and colors
    if src_vertices.size == 0 or src_colors.size == 0:
        raise RuntimeError("Source vertices or colors are empty")
    
    if len(src_vertices) != len(src_colors):
        raise RuntimeError("Source vertices and colors length mismatch")
    
    # Examines validity of destination mesh
    dst_vertices = np.asarray(dst_mesh.vertices)
    if dst_vertices.size == 0:
        raise RuntimeError("Destination mesh has no vertices")
    
    # Builds KD-Tree
    tree = cKDTree(src_vertices)

    # Queries all vertices
    _, idx = tree.query(dst_vertices, k=1)

    dst_colors = src_colors[idx]
    dst_mesh.vertex_colors = o3d.utility.Vector3dVector(dst_colors)