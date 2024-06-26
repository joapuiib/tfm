#!/usr/bin/env python3

import os
from PIL import Image
import re
import argparse
import pandas as pd
import sys
import numpy as np

Image.MAX_IMAGE_PIXELS = 260000000

class CropWithVariance:
    def __init__(self, images_csv, dest_dir, crop_size, shift):
        self.origin_dir = os.path.dirname(images_csv)
        self.dest_dir = dest_dir
        self.crop_size = crop_size
        self.shift = shift

        self.result_df = None

    def crop_and_discard(self, csv_images):
        self.load_df(csv_images)
        self.process_images()


    def load_df(self, csv_images):
        print(f'Loading dataframe from {csv_images}')
        self.df_images = pd.read_csv(csv_images)
        print(f'Loaded {len(self.df_images)} images')


    def process_images(self):
        # Loop through pandas dataframe
        for index, entry in self.df_images.iterrows():
            image_id = entry['image_id']
            top_label = entry['top_label_name']
            filename = os.path.join(self.origin_dir, top_label, image_id)

            exists = self.check_image_exists(filename)
            if exists:
                self.create_crops(filename)


    def check_image_exists(self, filename):
        return os.path.exists(filename)


    def create_crops(self, filename):
        img = Image.open(filename)
        crop_regions = self.compute_crop_regions(img)
        print(f'Computed {len(crop_regions)} crops for {filename}')

        for crop_region in crop_regions:
            crop_path = self.get_crop_path(filename, crop_region)
            crop_img = self.crop_image(img, crop_region)
            variance = self.compute_variance(crop_img)
            print(f'Computed variance {variance} for {crop_path}')
            self.save_crop(crop_img, variance)


    def compute_crop_regions(self, img):
        width, height = img.size

        crops = []

        x = 0
        y = 0
        end_of_image = False
        while not end_of_image:
            if y + self.crop_size >= height:
                y = height - self.crop_size
                end_of_image = True

            x = 0
            end_of_row = False
            while not end_of_row:
                if x + self.crop_size > width:
                    x = width - self.crop_size
                    end_of_row = True

                crops.append((x, y))
                x += self.shift

            y += self.shift
        return crops


    def get_crop_path(self, filename, crop_region):
        x, y = crop_region

        pattern = re.compile(r'(.+)(?:\(.*\))')
        image_prefix = re.match(pattern, filename).group(1)
        image_ext = os.path.splitext(filename)[1]
        coords = f"({x},{y},{x + self.crop_size},{y + self.crop_size})"
        new_image_path = f'{image_prefix}{coords}{image_ext}'

        return new_image_path



    def crop_image(self, img, crop_region):
        x, y = crop_region
        if x + self.crop_size > img.width or y + self.crop_size > img.height:
            raise ValueError('Crop region exceeds image dimensions')

        crop = img.crop((x, y, x + self.crop_size, y + self.crop_size))
        return crop


    def compute_variance(self, img):
        # Convert image to numpy array
        pixel_data = np.array(img)
        # Calculate variance for each channel and then take the mean variance
        variance = np.var(pixel_data, axis=(0, 1))
        mean_variance = np.mean(variance)
        return mean_variance


    def save_crop(self, crop_img, filename):
        pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Crop images and save variance')
    parser.add_argument('--dest_dir', help='Directory to save cropped images')
    parser.add_argument('--crop_size', type=int, help='Size of crop')
    parser.add_argument('--shift', type=int, help='Shift of crop')
    parser.add_argument('--csv_images', help='CSV file containing image filenames')
    args = parser.parse_args()

    crop_with_variance = CropWithVariance(args.csv_images, args.dest_dir, args.crop_size, args.shift)
    crop_with_variance.crop_and_discard(args.csv_images)
