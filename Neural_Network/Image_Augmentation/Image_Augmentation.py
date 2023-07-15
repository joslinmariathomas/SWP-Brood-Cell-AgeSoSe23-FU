import torch
import torchvision.transforms as transforms
import torch.nn as nn

resizedCellWidth = 64
resizedCellHeight = 64
pAugment = 0.8

augmentBase = [
    transforms.RandomApply(nn.ModuleList([transforms.RandAugment(num_ops=5)]), pAugment),
    transforms.RandomApply(nn.ModuleList([transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 0.5))]), pAugment),
    transforms.RandomApply(nn.ModuleList([transforms.RandomAdjustSharpness(sharpness_factor=2)]), pAugment),
    transforms.RandomApply(nn.ModuleList([transforms.RandomHorizontalFlip(p=0.6)]), pAugment),
    transforms.RandomApply(nn.ModuleList([transforms.RandomVerticalFlip(p=0.6)]), pAugment)
]

augment = nn.Sequential(*augmentBase, transforms.CenterCrop(resizedCellWidth))

def augment_image(image_cell):
    augmented_image_cell = augment(image_cell)
    return augmented_image_cell
