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


def training_tensor_gaussian_blur_save():
    training_tensor_folder =  "/content/SWP-Brood-Cell-AgeSoSe23-FU/training_tensor_data/tensors/"
    folder_to_save = "/content/SWP-Brood-Cell-AgeSoSe23-FU/Neural_Network/Gaussian_Blur/training_tensor_data/tensors"
    for file_name in os.listdir(training_tensor_folder):
        training_tensor = torch.load(
            f'{training_tensor_folder}/{file_name}')
        num_of_tensors = training_tensor.shape[0]
        appended_tensors = []
        for i in range(num_of_tensors):
            image_tensor = training_tensor[i, :, :]
            central_blurred_tensor = apply_gaussian_blur(image_tensor)
            appended_tensors.append(central_blurred_tensor)
            

        appended_tensor = torch.cat([tensor.unsqueeze(0) for tensor in appended_tensors], dim=0)
        torch.save(appended_tensor,
               f'{folder_to_save}/{file_name}')


def testing_tensor_gaussian_blur_save():
  testing_tensor_folder = "/content/SWP-Brood-Cell-AgeSoSe23-FU/testing_tensor_data/tensors/"
  folder_to_save = "/content/SWP-Brood-Cell-AgeSoSe23-FU/Neural_Network/Gaussian_Blur/testing_tensor_data/tensors"
  for file_name in os.listdir(testing_tensor_folder):
    testing_tensor = torch.load(
    f'{testing_tensor_folder}/{file_name}')
    num_of_tensors = testing_tensor.shape[0]
    appended_tensors = []
    for i in range(num_of_tensors):
      image_tensor = testing_tensor[i, :, :]
      central_blurred_tensor = apply_gaussian_blur(image_tensor)
      appended_tensors.append(central_blurred_tensor)

    appended_tensor = torch.cat([tensor.unsqueeze(0) for tensor in appended_tensors], dim=0)
    torch.save(appended_tensor,
    f'{folder_to_save}/{file_name}')


if __name__ == "__main__":
    training_tensor_gaussian_blur_save()
    testing_tensor_gaussian_blur_save()





