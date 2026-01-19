import os
import requests
import torch

import numpy as np

import matplotlib.pyplot as plt

from transformers import AutoImageProcessor, AutoModelForDepthEstimation, pipeline
from PIL import Image

def load_model():
    """
    A function that loads the DepthAnything V1 model,
    along with an image processor for pre- and post-processing
    of the input and output images.
    """

    checkpoints = [
    "Intel/zoedepth-nyu-kitti",
    "LiheYoung/depth-anything-large-hf",
    "jingheya/lotus-depth-g-v1-0",   
    "tencent/DepthCrafter"
    ]

    checkpoint = "LiheYoung/depth-anything-large-hf"

    image_processor = AutoImageProcessor.from_pretrained(checkpoint)
    model = AutoModelForDepthEstimation.from_pretrained(checkpoint).to("cpu")

    return checkpoint, image_processor, model

def run_depth_estimation(rgb_dir, depth_npy_dir, depth_bw_dir, depth_colored_dir, color_scheme="inferno"):
    """
    A function that batch-processes a given 
    dataset of images, in order to produce
    depth maps using an MDE model
    """

    # Loads the MDE model and image processors
    checkpoint, image_processor, model = load_model()

    # Initializes directories
    rgb_dir_sorted = sorted(os.listdir(rgb_dir))

    os.makedirs(depth_npy_dir, exist_ok=True)
    os.makedirs(depth_bw_dir, exist_ok=True)
    os.makedirs(depth_colored_dir, exist_ok=True)

    # Runs depth estimation algorithm 
    # on given dataset
    for filename in rgb_dir_sorted:
        if not filename.lower().endswith((".png", ".jpg", ".jpeg")):
            print("Folder did not contain images in valid input forms (.png, .jpg, .jpeg)")

        root = filename.split(".", 1)[0]

        img_path = os.path.join(rgb_dir, filename)

        image = Image.open(img_path).convert("RGB")

        inputs = image_processor(images=image, return_tensors="pt")

        with torch.no_grad():
            outputs = model(**inputs)

        post_processed_output = image_processor.post_process_depth_estimation(
            outputs,
            target_sizes=[(image.height, image.width)]
        )

        predicted_depth = post_processed_output[0]["predicted_depth"]

        # Save .npy depth map
        npy_path = os.path.join(depth_npy_dir, root + "_depth.npy")

        np.save(npy_path, predicted_depth.cpu().numpy())

        # Save grayscale depth map
        depth_bw = (predicted_depth - predicted_depth.min()) / (predicted_depth.max() - predicted_depth.min())
        depth = (depth_bw.cpu().numpy() * 255).astype("uint8")

        bw_path = os.path.join(depth_bw_dir, root + "_depth_bw.png")
        Image.fromarray(depth).save(bw_path)

        # Save colored depth map
        colormap = plt.colormaps[color_scheme]

        depth_colored = colormap(depth_bw.cpu().numpy())
        depth_colored = (depth_colored[:,:,:3] * 255).astype("uint8")

        colored_path = os.path.join(depth_colored_dir, root + "_depth_colored.png")
        Image.fromarray(depth_colored).save(colored_path)

        print(f"Saved {root}!")
    
    print("Done!")

if __name__ == "__main__":
    rgb_dir = "data"

    depth_npy_dir = "depth_npy"
    depth_bw_dir = "depth_bw"
    depth_colored_dir = "depth_colored"

    color_scheme = "inferno"

    run_depth_estimation(rgb_dir=rgb_dir, depth_npy_dir=depth_npy_dir, depth_bw_dir=depth_bw_dir, depth_colored_dir=depth_colored_dir, color_scheme=color_scheme)
