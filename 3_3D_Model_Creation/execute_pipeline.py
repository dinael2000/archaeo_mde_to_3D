from openback import *
from watertight import *
from merge_meshes import *
from scale_meshes import *

# Or just run from each individual file. 
# Make sure that the directories are set up correctly!

option = int(input("What do you want to create?:\n 1. A simple mesh (uncropped)\n 2. An open-back mesh\n 3. A watertight mesh\n 4. A merged mesh\n 5. Scale a mesh\n"))

# Set up directories
depth_dir = r"1_DA1_Depth_Estimation/result_depth_maps/depth_npy"
rgb_dir = r"data"

if option == 1:
    out_dir = r"3_3D_Model_Creation/3d_models_uncropped"
    batch_process_no_cropping(depth_dir=depth_dir, rgb_dir=rgb_dir, output_dir=out_dir)

if option == 2:
    out_dir = r"3_3D_Model_Creation/3d_models_openback"
    batch_process_openback(depth_dir, rgb_dir, out_dir, "obj", pixel_size=1.0, relief_scale=7.0, poisson_depth=9)

if option == 3:
    out_dir = r"3_3D_Model_Creation/3d_models_watertight"
    batch_process_watertight(depth_dir, rgb_dir, out_dir, "obj", pixel_size=1.0, relief_scale=7.0, poisson_depth=9)

if option == 4:
    input_dir = r"3_3D_Model_Creation/3d_models_openback"
    output_dir = r"3_3D_Model_Creation/3d_models_merged"
    batch_process_merge(input_dir, output_dir, gap_factor=0.0, scale=True, scale_factor=0.25)

if option == 5:
    input_dir = r"3_3D_Model_Creation/3d_models_merged"
    output_dir = r"3_3D_Model_Creation/3d_models_merged_scaled"
    scale_factor = 0.02


    batch_scale_models(input_dir, output_dir, scale_factor)
