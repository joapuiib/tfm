#!/usr/bin/env python3

import os
from PIL import Image
import re
import argparse
import json
import sys

parser = argparse.ArgumentParser(description='Find related images in two directories')
parser.add_argument('origin_dir', type=str, help='The directory containing the original images')
parser.add_argument('dest_dir', type=str, help='The directory containing the images to compare')
args = parser.parse_args()

dir1 = os.path.expanduser(args.origin_dir)
dir2 = os.path.expanduser(args.dest_dir)

dir1_files = {}

# Create the destination directory if it doesn't exist

pattern = re.compile(r'(.+)(?:\(.*\))')

# Walk through the dir1 directory
for root, _, files in os.walk(dir1):
    for file in files:
        if file.endswith(".png"):
            filename = os.path.join(root, file)
            relative_path = os.path.relpath(filename, dir1)
            image_prefix = re.match(pattern, relative_path).group(1)

            dir1_files[image_prefix] = []

for root, _, files in os.walk(dir2):
    for file in files:
        if file.endswith(".png"):
            filename = os.path.join(root, file)
            relative_path = os.path.relpath(filename, dir2)
            image_prefix = re.match(pattern, relative_path).group(1)

            try:
                dir1_files[image_prefix].append(relative_path)
            except KeyError:
                print(f"Image {image_prefix} not found in {dir1}", file=sys.stderr)

print(json.dumps(dir1_files, indent=4))
