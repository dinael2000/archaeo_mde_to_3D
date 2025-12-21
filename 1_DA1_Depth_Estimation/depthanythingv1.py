import os
import requests
import torch

import numpy as np

import matplotlib.pyplot as plt

from transformers import AutoImageProcessor, AutoModelForDepthEstimation, pipeline
from PIL import Image

rgb_dir = "data"

depth_npy_dir = "1_DA1_Depth_Estimation/results_depth_maps/depth_npy"
depth_colored_dir = "1_DA1_Depth_Estimation/results_depth_maps/depth_colored"
depth_bw_dir = "1_DA1_Depth_Estimation/results_depth_maps/depth_bw"

os.makedirs(depth_npy_dir, exist_ok=True)
os.makedirs(depth_colored_dir, exist_ok=True)
os.makedirs(depth_bw_dir, exist_ok=True)

checkpoints = [
    "Intel/zoedepth-nyu-kitti",
    "LiheYoung/depth-anything-large-hf",
    "jingheya/lotus-depth-g-v1-0",   
    "tencent/DepthCrafter"
    ]

checkpoint = "LiheYoung/depth-anything-large-hf"

# Loads models
image_processor = AutoImageProcessor.from_pretrained(checkpoint)
model = AutoModelForDepthEstimation.from_pretrained(checkpoint).to("cpu")

# Inference loop
for filename in sorted(os.listdir(rgb_dir)):
    if not filename.lower().endswith((".png", ".jpg", ".jpeg")):
        continue

    base = filename.rsplit(".", 1)[0]

    img_path = os.path.join(rgb_dir, filename)

    image = Image.open(img_path).convert("RGB")

    # Preprocesses
    inputs = image_processor(images=image, return_tensors="pt")

    # Predicts
    with torch.no_grad():
        outputs = model(**inputs)

    # Post-processes
    post_processed_output = image_processor.post_process_depth_estimation(
        outputs,
        target_sizes=[(image.height, image.width)]
    )

    predicted_depth = post_processed_output[0]["predicted_depth"]

    # Saves .npy
    npy_path = os.path.join(depth_npy_dir, base + "_depth.npy")
    np.save(npy_path, predicted_depth.cpu().numpy())

    # Saves grayscale depth .png
    depth_norm = (predicted_depth - predicted_depth.min()) / (predicted_depth.max() - predicted_depth.min())
    depth = (depth_norm.cpu().numpy() * 255).astype("uint8")

    gray_path = os.path.join(depth_bw_dir, base + "_depth.png")

    Image.fromarray(depth).save(gray_path)

    # Saves colored depth .png

    colormap = plt.colormaps["inferno"]

    depth_colored = colormap(depth_norm.cpu().numpy())

    depth_colored = (depth_colored[:,:,:3] * 255).astype("uint8")

    colored_path = os.path.join(depth_colored_dir, base + "_depth_colored.png")

    Image.fromarray(depth_colored).save(colored_path)

    print(f"Saved {npy_path} and {gray_path}")