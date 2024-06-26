#!/usr/bin/env python3

from PIL import Image
import os
import numpy as np
import argparse

def image_variance(image_path):
    """
    Calculates the variance of an image's pixel values across all channels.
    """
    with Image.open(image_path) as img:
        # Convert image to numpy array
        pixel_data = np.array(img)
        # Calculate variance for each channel and then take the mean variance
        variance = np.var(pixel_data, axis=(0, 1))
        mean_variance = np.mean(variance)
        return mean_variance

def discard_image_if_low_variance(image_path, threshold):
    """
    Discards images from a directory whose variance is below a certain threshold.
    """
    variance = image_variance(image_path)
    print(f"Variance for {image_path}: {variance}")
    if variance < threshold:
        os.rename(image_path, image_path + ".discarded")
        print(f"Discarded")


def discard_images_in_directory(origin_dir, threshold):
    # Walk through the origin directory and find all .png files
    for root, _, files in os.walk(origin_dir):
        for file in files:
            if file.endswith(".png"):
                # Construct the full file path
                filename = os.path.join(root, file)

                # Discard the image if its variance is below the threshold
                discard_image_if_low_variance(filename, threshold)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Discard images with low variance")
    parser.add_argument("image_dir", help="Directory containing images")
    parser.add_argument("threshold", type=float, help="Variance threshold")
    args = parser.parse_args()

    discard_images_in_directory(args.image_dir, args.threshold)
