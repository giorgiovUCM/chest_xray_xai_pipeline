import cv2
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import models, transforms

import sys
sys.path.append('.')
from config import IMAGENET_MEAN, IMAGENET_STD


"""
Shared utility functions and classes for the chest X-ray XAI pipeline.

Includes:
- apply_nclahe: N-CLAHE preprocessing for chest X-ray images
- ChestXrayDataset: PyTorch Dataset class for VinDr-CXR multi-label classification
- build_model: builds AlexNet or DenseNet-121 for inference (weights=None)

Note: normalization parameters (mean and std) correspond to ImageNet statistics,
as both architectures were pretrained on ImageNet.
"""


def apply_nclahe(image_np, tile_size):
    """
    Applies N-CLAHE preprocessing to a grayscale image.
    Performs a logarithmic normalization prior to CLAHE to linearize
    the exponential nature of X-ray pixel intensities (Beer-Lambert law).
 
    Args:
        image_np: numpy array of grayscale image
        tile_size: CLAHE tile size (4 for 256x256, 16 for 1024x1024)
 
    Returns:
        Preprocessed grayscale image as uint8 numpy array
    """
    image_log = np.log1p(image_np.astype(np.float32))
    image_log = ((image_log - image_log.min()) /
                  (image_log.max() - image_log.min()) * 255).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(tile_size, tile_size))
    return clahe.apply(image_log)


class ChestXrayDataset(Dataset):
    """
    PyTorch Dataset class for chest X-ray radiographs from VinDr-CXR.

    Applies N-CLAHE preprocessing and ImageNet normalization.
    Supports multi-label classification for aortic enlargement (class_id=0)
    and cardiomegaly (class_id=1).

    Args:
        metadata_csv: path to metadata CSV with image_id and class_id columns
        split_csv: path to CSV with image_ids for the split (train/val/test)
        images_dir: directory containing PNG images
        resolution: image resolution (256 or 1024)
        is_for_train: if True applies data augmentation (rotation and random crop)

    Returns:
        img: preprocessed and normalized image tensor
        label: multi-label binary tensor [aneurysm, cardiomegaly]
        image_id: string identifier of the image
    """
    def __init__(self, metadata_csv, split_csv, images_dir, resolution, is_for_train=False):
        self.df = pd.read_csv(metadata_csv)
        self.split_ids = pd.read_csv(split_csv)['image_id'].tolist()
        self.df = self.df[self.df['image_id'].isin(self.split_ids)]
        self.images_dir = images_dir
        self.resolution = resolution
        self.is_for_train = is_for_train
        self.tile_size = 4 if resolution == 256 else 16
        self.image_ids = self.df['image_id'].unique().tolist()
        self.normalize = transforms.Normalize(
            mean=IMAGENET_MEAN,
            std=IMAGENET_STD
        )
        self.augment = transforms.Compose([
            transforms.RandomRotation(degrees=5),
            transforms.RandomResizedCrop(resolution, scale=(0.9, 1.0)),
        ]) if is_for_train else None

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        img = cv2.imread(f'{self.images_dir}/{image_id}.png', cv2.IMREAD_GRAYSCALE)
        img = apply_nclahe(img, self.tile_size)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        img = Image.fromarray(img)
        if self.is_for_train and self.augment:
            img = self.augment(img)
        img = transforms.ToTensor()(img)
        img = self.normalize(img)
        rows = self.df[self.df['image_id'] == image_id]
        label = torch.zeros(2)
        if 0 in rows['class_id'].values:
            label[0] = 1.0
        if 1 in rows['class_id'].values:
            label[1] = 1.0
        return img, label, image_id


def build_model(architecture, device): 
    """
    Builds AlexNet or DenseNet-121 for multi-label classification.
    Replaces the final classifier with a 2-output linear layer.
 
    Args:
        architecture: 'alexnet' or 'densenet'
        device: torch device ('cuda' or 'cpu')
 
    Returns:
        model moved to device
    """

    if architecture == 'alexnet':
        model = models.alexnet(weights=None)
        model.classifier[6] = nn.Linear(4096, 2)
    elif architecture == 'densenet':
        model = models.densenet121(weights=None)
        model.classifier = nn.Linear(1024, 2)
    return model.to(device)
