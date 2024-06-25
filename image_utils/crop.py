#!/usr/bin/env python3

import os
from PIL import Image
import re

Image.MAX_IMAGE_PIXELS = 260000000

origin_dir = "7000/"
dest_dir = "800_custom/"
crop_size = 1812
shift = 1756

def create_crops(image_path, crop_size, shift):
    # Open the image
    img = Image.open(image_path)
    img_width, img_height = img.size
    print(f'Image: {image_path}')
    print(f'Image size: {img_width}x{img_height}')

    # Determine the destination filename
    relative_path = os.path.relpath(filename, origin_dir)
    dest_filename = os.path.join(dest_dir, relative_path)

    # Create the necessary directories in the destination path
    base_dest_dir = os.path.dirname(dest_filename)
    os.makedirs(base_dest_dir, exist_ok=True)

    pattern = re.compile(r'(.+)(?:\(.*\))')
    image_prefix = re.match(pattern, dest_filename).group(1)
    image_ext = os.path.splitext(dest_filename)[1]
    print(f'Image prefix: {image_prefix}')
    print(f'Image extension: {image_ext}')

    num_crops = 0

    x = 0
    y = 0
    end_of_image = False
    while not end_of_image:
        if y + crop_size >= img_height:
            y = img_height - crop_size
            end_of_image = True

        x = 0
        end_of_row = False
        while not end_of_row:

            if x + crop_size > img_width:
                x = img_width - crop_size
                end_of_row = True

            # print(f'Creating crop at ({x}, {y}, {x + crop_size}, {y + crop_size})')
            num_crops += 1

            # Crop the image
            crop = img.crop((x, y, x + crop_size, y + crop_size))
            coords = f"({x},{y},{x + crop_size},{y + crop_size})"
            new_image_path = f'{image_prefix}{coords}{image_ext}'
            print(f'Saving crop to {new_image_path}')
            crop.save(new_image_path)

            x += shift

        y += shift

    print(f'Created {num_crops} crops')

origin_dir = os.path.expanduser(origin_dir)
dest_dir = os.path.expanduser(dest_dir)

# Create the destination directory if it doesn't exist
os.makedirs(dest_dir, exist_ok=True)
print(f"Resizing images from {origin_dir} and saving them to {dest_dir}")

if not os.path.exists(origin_dir):
    print(f"Origin directory {origin_dir} does not exist.")
    exit(1)


# Walk through the origin directory and find all .png files
for root, _, files in os.walk(origin_dir):
    for file in files:
        if file.endswith(".png"):
            # Construct the full file path
            filename = os.path.join(root, file)

            # Open the image, resize it, and save it to the destination
            create_crops(filename, crop_size, shift)
