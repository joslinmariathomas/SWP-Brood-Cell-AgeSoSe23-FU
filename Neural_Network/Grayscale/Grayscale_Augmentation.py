import random
import torchvision.transforms as T
import torch
import os
import random
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter

def apply_gaussian_blur(image_tensor):
    blur_amount, blur_radius = 2.0,20
    central_blur_amount, central_blur_radius = 0.9,7
    # Convert the tensor to a numpy array
    image_np = image_tensor.permute(1, 2, 0).numpy()

    # Create a circular mask for the surrounding region
    h, w, _ = image_np.shape
    center_x, center_y = w // 2, h // 2
    y, x = np.ogrid[:h, :w]
    mask = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2) > blur_radius

    # Apply Gaussian blur to the surroundings
    blurred_np = np.copy(image_np)
    blurred_np[mask] = gaussian_filter(image_np, sigma=blur_amount)[mask]

    # Create a circular mask for the central region
    central_mask = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2) <= central_blur_radius

    # Apply a lighter Gaussian blur to the central region
    central_blurred_np = np.copy(blurred_np)
    central_blurred_np[central_mask] = gaussian_filter(image_np, sigma=central_blur_amount)[central_mask]

    # Convert the blurred numpy array back to a tensor
    central_blurred_tensor = torch.from_numpy(central_blurred_np).permute(2, 0, 1)

    return central_blurred_tensor


def augment_image(image_cell, probability: float):
    # augmented_tensor = apply_gaussian_blur(image_cell)
    augmented_tensor = image_cell
    if random.random() < probability:
        augmented_tensor = T.GaussianBlur(kernel_size=3, sigma=0.75)(
            augmented_tensor)

    if random.random() < probability:
        augmented_tensor = T.RandomAdjustSharpness(sharpness_factor=2)(
            augmented_tensor)

    if random.random() < probability:
        augmented_tensor = T.RandomHorizontalFlip()(augmented_tensor)

    if random.random() < probability:
        augmented_tensor = T.RandomVerticalFlip()(augmented_tensor)

    augmented_tensor = T.Grayscale()(augmented_tensor)

    return augmented_tensor
