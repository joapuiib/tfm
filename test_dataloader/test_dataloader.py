#!/usr/bin/env python

import torch
import pandas as pd
from torchvision.io import read_image
from torchvision.transforms import functional as TF
from torch.utils.data import Dataset, DataLoader
import os
import matplotlib.pyplot as plt
from math import sqrt, ceil

class TestDataset(Dataset):
    def __init__(self, annotations_file, img_dir, patch_height, patch_width, vertical_offset, horizontal_offset):
        self.img_labels = pd.read_csv(annotations_file, header=None)
        self.img_dir = img_dir
        self.patch_height = patch_height
        self.patch_width = patch_width
        self.vertical_offset = vertical_offset
        self.horizontal_offset = horizontal_offset

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        """
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            print("worker {} loading index {}".format(worker_info.id, idx))
        """

        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        # print(img_path)
        image = read_image(img_path)
        image = self.extract_patches(image)

        label = self.img_labels.iloc[idx, 1]
        return image, label

    def extract_patches(self, image):
        """
        Extract patches from a given image tensor.

        Parameters:
        - image: A torch tensor representing the image, with shape (C, H, W).
        - patch_height: The height of each patch.
        - patch_width: The width of each patch.
        - vertical_offset: The vertical offset between patches.
        - horizontal_offset: The horizontal offset between patches.

        Returns:
        - A tensor containing all the patches extracted from the image, with shape (N, C, patch_height, patch_width),
          where N is the number of patches.
        """
        C, H, W = image.shape
        patches = []

        if H < self.patch_height or W < self.patch_width:
            raise ValueError("The image is smaller than the patch size")

        for i in range(0, H - self.patch_height + 1, self.vertical_offset):
            for j in range(0, W - self.patch_width + 1, self.horizontal_offset):
                patch = image[:, i:i+self.patch_height, j:j+self.patch_width]
                patches.append(patch.unsqueeze(0))

        return torch.cat(patches, dim=0)

def collate_fn(batch):
    """
    Collate function to be used with the DataLoader.
    It takes a batch of patched images (4D) and returns
    a 3D tensor with each patch labeled.

    For each sample, the batch contains the patches extracted from the image (with shape (N, C, H, W)) and the label.

    The method returns a single tensor containg all the patches of the batch, with shape (N, C, H, W), and a tensor
    containing the labels of the batch.
    """
    patches = []
    labels = []
    for image, label in batch:
        for patch in image:
            patches.append(patch)
            labels.append(label)

    patches = torch.stack(patches)
    # print(patches.shape)

    patches_labels = torch.tensor(labels)
    # print(patches_labels.shape)
    return patches, patches_labels

if __name__ == "__main__":
    dataset = TestDataset(
            annotations_file='data/labels.csv',
            img_dir='data/',
            patch_height=100,
            patch_width=100,
            vertical_offset=100,
            horizontal_offset=100
    )
    dataloader = DataLoader(
        dataset,
        batch_size=2,
        num_workers=1,
        collate_fn=collate_fn,
        shuffle=True,
    )

    print("Number of samples in the dataset: ", len(dataloader))

    for (patches, labels) in dataloader:
        sqrt_size = ceil(sqrt(len(patches)))
        fig = plt.figure(figsize=(sqrt_size, sqrt_size))
        # plt.text(5, 5, f"class: {label}", bbox={'facecolor': 'white', 'pad': 10})
        for i, (patch, label) in enumerate(zip(patches, labels)):
            fig.add_subplot(sqrt_size, sqrt_size, i+1)
            plt.title(f"class: {label}", fontsize=8)
            plt.imshow(patch.permute(1, 2, 0))
            # plt.imshow(img)
        plt.subplots_adjust(hspace=0.75)
        plt.show()
