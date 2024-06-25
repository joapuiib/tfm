#!/usr/bin/env python3

import os
from PIL import Image

origin_dir = "800/"
dest_dir = "800_224/"
dest_size_width = 224
dest_size_height = 224

modes = {
    "LANCZOS": Image.LANCZOS,
    # "NEAREST": Image.NEAREST,
    # "BICUBIC": Image.BICUBIC,
    # "BILINEAR": Image.BILINEAR,
    # "HAMMING": Image.HAMMING,
    # "BOX": Image.BOX
}

for name, resize_mode in modes.items():
    # dest_dir = f"~/images/7000_1812_{name}/"

    origin_dir = os.path.expanduser(origin_dir)
    dest_dir = os.path.expanduser(dest_dir)

    # Create the destination directory if it doesn't exist
    os.makedirs(dest_dir, exist_ok=True)
    print(f"Resizing images from {origin_dir} and saving them to {dest_dir}")

    if not os.path.exists(origin_dir):
        print(f"Origin directory {origin_dir} does not exist.")
        exit(1)

    Image.MAX_IMAGE_PIXELS = 252000000


    # Walk through the origin directory and find all .png files
    for root, _, files in os.walk(origin_dir):
        for file in files:
            if file.endswith(".png"):
                # Construct the full file path
                filename = os.path.join(root, file)
                
                # Determine the destination filename
                relative_path = os.path.relpath(filename, origin_dir)
                dest_filename = os.path.join(dest_dir, relative_path)

                print(f"Resizing {filename} and saving it to {dest_filename}...")
                
                # Create the necessary directories in the destination path
                base_dest_dir = os.path.dirname(dest_filename)
                os.makedirs(base_dest_dir, exist_ok=True)
                
                # Open the image, resize it, and save it to the destination
                with Image.open(filename) as img:
                    img = img.resize((dest_size_width, dest_size_height), resize_mode)
                    img.save(dest_filename, format='PNG', optimize=True)
