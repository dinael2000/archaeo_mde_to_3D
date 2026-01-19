import os

import open3d as o3d

def scale_models(mesh, scale_factor, output_path_scaled):

    mesh.scale(scale_factor, center=mesh.get_center())
    mesh.orient_triangles()
    mesh.compute_vertex_normals()

    o3d.io.write_triangle_mesh(output_path_scaled, mesh)

def batch_scale_models(input_dir, output_dir, scale_factor):
    os.makedirs(output_dir, exist_ok=True)

    files = os.listdir(input_dir)

    for file in files:
        stem = file.replace(".obj", "")

        input_path = os.path.join(input_dir, file)

        mesh = o3d.io.read_triangle_mesh(input_path)

        output_path_scaled = os.path.join(
            output_dir, f"{stem}-merged_scaled.obj"
        )

        scale_models(mesh, scale_factor=scale_factor, output_path_scaled=output_path_scaled)


if __name__ == "__main__":
    input_dir = r"3_3D_Model_Creation/3d_models_merged_19122025"
    output_dir = r"3_3D_Model_Creation/3d_models_merged_scaled_21122025"
    scale_factor = 0.02

    batch_scale_models(input_dir, output_dir, scale_factor)