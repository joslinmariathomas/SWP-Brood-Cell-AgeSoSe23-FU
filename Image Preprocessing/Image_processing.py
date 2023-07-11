import json
from torch.utils.data import Dataset, DataLoader, Subset
import uuid
import os
import torch
import numpy as np
import torch
import torchvision.transforms as transforms
from torchvision.transforms import functional as F
from skimage import io
import matplotlib.pyplot as plt
import torch.nn as nn
from helper_functions import (import_from_json, replace_image_extension)
import math
from datetime import datetime


def rotate(origin, point, angle):
    """
    Rotate a point counterclockwise by a given angle around a given origin.

    The angle should be given in radians.
    """
    ox, oy = origin
    px, py = point

    qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
    qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
    return qx, qy


def imshow(img, normalize=True):
    if normalize:
        img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    fig = plt.figure(figsize=(30, 30))
    ax = fig.add_subplot(111)
    ax.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def flatten(l):
    return [item for sublist in l for item in sublist]


def transform_image(image):
    rotate_degrees = 1.0
    imagePadding = 60
    if image.shape[0] == 4:
        image = image[:3, :, :]

    # Convert tensor image to PIL image
    pil_image = F.to_pil_image(image)

    # Rotate the PIL image
    rotated_pil_image = F.rotate(pil_image, rotate_degrees)

    # Convert the rotated PIL image back to a tensor
    tensor_image = F.to_tensor(rotated_pil_image)

    # Pad the image
    padded_image = F.pad(tensor_image, (
    imagePadding, imagePadding, imagePadding, imagePadding), fill=0)

    return padded_image


def read_image(filename, f_ending, save=None):
    try:
        if f_ending == "tiff":
            raw_image = plt.imread(filename)
        else:
            raw_image = torch.from_numpy(io.imread(filename))
    except Exception as e:
        print(e)
        return None

    image = transform_image(raw_image.permute(2, 0, 1)).permute(1, 2,
                                                                0).double()
    return image


def getCellIndices():
    indices = []
    numCellsY = 18
    numCellsX = 16
    offsetY = 248
    offsetX = 288
    evenRowOffsetX = 150
    centerOffsetX = 300
    centerOffsetY = 510
    cellHeight = 400
    cellWidth = 400

    for y in range(numCellsY):
        for x in range(numCellsX):
            startY = y * offsetY + centerOffsetY - cellHeight // 2
            startX = (
                    (0 if y % 2 == 0 else evenRowOffsetX)
                    + x * offsetX
                    + centerOffsetX
                    - cellWidth // 2
            )
            indices.append((((startX, startX + cellWidth),
                             (startY, startY + cellHeight)), (x, y)))
    return indices


def getCellsFromImage(image, indices, json_image_cells):
    cells = []
    resizedCellWidth = 64
    resizedCellHeight = 64

    for ((startX, endX), (startY, endY)), (x, y) in indices:
        cell_indices = (
        transforms.functional.resize(image[startY: endY, startX: endX].permute(
            2, 0, 1), (resizedCellWidth, resizedCellHeight)),
        ((startX, endX), (startY, endY)), (x, y))
        cell_id = f"cell_id_{str(uuid.uuid4())}"
        cells.append(
            {"cell_coordinates": cell_indices[2],
             "cell_id": cell_id,
             "cell_image": cell_indices[0].tolist(),
             "cell_indices": cell_indices[1]}
        )
        json_image_cells[y][x]["cell_id"] = cell_id
    return cells


def main():
    jsonfile = import_from_json(filename="../full_dataset_predictions.json")
    json_file_extension_updated = replace_image_extension(jsonfile)
    folder_path = './Images'
    available_images = os.listdir(folder_path)
    file_counter = 0
    available_training_json = [frame for frame in json_file_extension_updated
                               if
                               frame[
                                   "filename"] in available_images]
    for filename in os.listdir(folder_path):
        filename_without_extension = os.path.splitext(filename)[0]
        for json_image_labels in available_training_json:
            if json_image_labels["filename"] == filename:
                image = read_image(f"{folder_path}/{filename}",
                                   f_ending=".png", save=None)
                cell_indices = getCellIndices()
                cells = getCellsFromImage(image, cell_indices,
                                          json_image_labels["cells"])

                with open(
                        f"./cells_image_data/{filename_without_extension}.json",
                        'w') as f:
                    json.dump(cells, f)
                file_counter = file_counter + 1
    with open("../full_dataset_predictions_updated.json", 'w') as f:
        json.dump(available_training_json, f)


if __name__ == '__main__':
    main()

# # image = read_image("./Images/scan_back_220810-044352-utc.png", f_ending=".png", )
# image = import_from_json("./cells_image_data/scan_back_220810-134000-utc.json")
# image = image[0].get("cell_image")
# # cell_indices = getCellIndices()
# # cells = getCellsFromImage(image, cell_indices)
# # (cell1,_,_) = cells[-1]
# #
# image = torch.tensor(image)
# cell1 = image.permute(1,2,0)
#
#
# # image = torch.from_numpy(io.imread("./Images/scan_back_220810-044352-utc.png"))
# image_array = np.array(cell1)
# # image_array = cell1.numpy()
#
# # Display the image
# plt.imshow(image_array)
# plt.axis('off')
# plt.show()
# print(image.shape)
