#!/usr/bin/env python3

import os
from PIL import Image
import re
import argparse

Image.MAX_IMAGE_PIXELS = 260000000

class CropDiscard:
    def __init__(self, origin_dir, dest_dir, crop_size, shift, min_variance):
        self.origin_dir = origin_dir
        self.dest_dir = dest_dir
        self.crop_size = crop_size
        self.shift = shift
        self.min_variance = 1000

        self.result_df = None

    def crop_and_discard(self, csv_images):
        load_df(csv_images)
        process_images()


    def load_df(self, csv_images):
        print(f'Loading dataframe from {csv_images}')
        self.df_images = pd.read_csv(csv_images)


    def process_images(self):
        for filename in self.df_images['filename']:
            self.create_crops(filename)


    def create_crops(self, filename):
        img = Image.open(filename)
        crop_regions = self.compute_crop_regions(img)
        print(f'Computed {len(crop_regions)} crops for {filename}')

        for crop_region in crop_regions:
            crop_img = self.crop_image(img, crop_region)
            variance = self.compute_variance(crop_img)
            if variance >= self.min_variance:
                self.save_crop(crop_img, crop_region, filename)


    def compute_crop_regions(self, img):
        width, height = img.size

        crops = []

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

                crops.append((x, y))

                x += shift

            y += shift


    def crop_image(self, img, crop_region):
        x, y = crop_region
        crop = img.crop((x, y, x + crop_size, y + crop_size))
        return crop


    def compute_variance(img):
        # Convert image to numpy array
        pixel_data = np.array(img)
        # Calculate variance for each channel and then take the mean variance
        variance = np.var(pixel_data, axis=(0, 1))
        mean_variance = np.mean(variance)
        return mean_variance


    def save_crop(crop_img, filename):
        pass
