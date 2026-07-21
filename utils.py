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


def parse_dicom_age(age_str):
    """Parses a DICOM-style age string (e.g. '045Y') into an integer number
    of years. Non-'Y' units (months/weeks/days) are treated as unknown
    rather than coerced to 0, since VinDr-CXR is an adult-only cohort and
    a 0-year age is not a meaningful value here."""
    if pd.isna(age_str):
        return np.nan
    age_str = str(age_str).strip()
    if len(age_str) < 4:
        return np.nan
    value, unit = age_str[:3], age_str[3]
    try:
        value = int(value)
    except ValueError:
        return np.nan
    if unit != 'Y':
        return np.nan
    return value


def age_to_group(age):
    """Maps a patient age (int) to a clinical age-group bucket: 18-39,
    40-64, 65+. Ages outside the plausible adult range (VinDr-CXR excludes
    pediatric scans) are treated as unknown, since implausible values like
    238 are DICOM encoding artifacts, not real ages."""
    if age is None or (isinstance(age, float) and np.isnan(age)):
        return 'unknown'
    age = int(age)
    if age < 18 or age > 100:
        return 'unknown'
    elif age <= 39:
        return '18-39'
    elif age <= 64:
        return '40-64'
    else:
        return '65+'
    

def pad_to_square(tensor):
    """Zero-pads a (C, H, W) tensor to square if it isn't already one.
    A no-op for this pipeline's images (already square), kept explicit so the
    documented preprocessing order (N-CLAHE -> Z-score -> zero padding ->
    augmentation) is literal in code."""
    c, h, w = tensor.shape
    if h == w:
        return tensor
    diff = abs(h - w)
    pad_a, pad_b = diff // 2, diff - diff // 2
    padding = (pad_a, pad_b, 0, 0) if h > w else (0, 0, pad_a, pad_b)
    return nn.functional.pad(tensor, padding, mode='constant', value=0.0)


def saliency_entropy(gradcam_map):
    """Shannon entropy of a normalized Grad-CAM map. Only comparable within
    the same architecture (AlexNet vs DenseNet have different native
    Grad-CAM grid sizes before upsampling)."""
    p = gradcam_map / (gradcam_map.sum() + 1e-8)
    return float(-np.sum(p * np.log(p + 1e-8)))


def compute_mmd(X, Y, gamma=1.0):
    """MMD (RBF kernel) between two sets of penultimate-layer feature vectors.
    High MMD desirable for pathology-vs-control or
    cardiomegaly-vs-aortic-enlargement; low MMD desirable across demographic
    strata within one class."""
    from sklearn.metrics.pairwise import rbf_kernel
    XX = rbf_kernel(X, X, gamma)
    YY = rbf_kernel(Y, Y, gamma)
    XY = rbf_kernel(X, Y, gamma)
    return float(XX.mean() + YY.mean() - 2 * XY.mean())


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
            transforms.RandomApply([transforms.RandomRotation(degrees=5)], p=0.5),
            transforms.RandomApply(
                [transforms.RandomResizedCrop(resolution, scale=(0.9, 1.0))], p=0.5
            ),
        ]) if is_for_train else None

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        img = cv2.imread(f'{self.images_dir}/{image_id}.png', cv2.IMREAD_GRAYSCALE)
        img = apply_nclahe(img, self.tile_size)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        img = Image.fromarray(img)
        img = transforms.ToTensor()(img)
        img = self.normalize(img)
        img = pad_to_square(img)
        if self.is_for_train and self.augment:
            img = self.augment(img)

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

def get_stratification(row, label):
    """
    Returns the prediction type (TP, FP, FN, TN) for a given label.
    
    Args:
        row: DataFrame row with pred_{label} and label_{label} columns
        label: 'aneurysm' or 'cardiomegaly'
    
    Returns:
        str: 'TP', 'FP', 'FN' or 'TN'
    """
    pred = row[f'pred_{label}']
    gt   = row[f'label_{label}']
    if pred == 1 and gt == 1:   return 'TP'
    elif pred == 0 and gt == 1: return 'FN'
    elif pred == 1 and gt == 0: return 'FP'
    else:                       return 'TN'