import os
import json
import uuid

import torch
import numpy as np
import matplotlib.pyplot as plt

from  imageio.v2 import imread
from torchvision import transforms
from torchvision.transforms import functional

from helper_functions import (
    import_from_json,
    replace_image_extension
)


class Imageprocessor:
    defaultCellSize = 400
    resizedCellWidth = 64
    resizedCellHeight = 64
    numCellsY = 18
    numCellsX = 16
    offsetY = 248
    offsetX = 288
    evenRowOffsetX = 150
    centerOffsetX = 300
    centerOffsetY = 510
    imageWidth = 4992
    imageHeight = imageWidth
    cellPadding = 0
    imagePadding = 60
    splitX = int(imageWidth * 0.4)
    splitX2 = int(imageWidth * 0.6)
    images = []
    trainIndices = []
    testIndices = []
    pAugment = 0.8
    num_images = -1
    cellSize = 400
    rotate_degrees = 1.0

    def __init__(self,
                 augmentTest: bool,
                 augmentTrain: bool,
                 augmentOriginal: bool):

        self.cellWidth = self.cellSize
        self.cellHeight = self.cellSize
        self.resultCellWidth = int(
            self.resizedCellWidth * (self.cellWidth / self.defaultCellSize))
        self.resultCellHeight = int(
            self.resizedCellHeight * (self.cellHeight / self.defaultCellSize))
        self.num_images = self.num_images
        self.image_map = {}
        self.augmentTest = augmentTest
        self.augmentTrain = augmentTrain
        self.augmentOriginal = augmentOriginal

    def read_image(self, filename, f_ending, save=None):
        # print(f'reading image {filename}...')
        try:
            if f_ending == "tiff":
                # raw_image = np.array(Image.open(img_name))
                raw_image = plt.imread(filename)
            else:
                raw_image = torch.from_numpy(imread(filename))
        except Exception as e:
            print(e)
            return None
        image = self.transform_image(raw_image.permute(2, 0, 1)).permute(1, 2,
                                                                         0).double()
        if save is not None:
            self.image_map[save] = image
        return image

    def transform_image(self, image):
        return self.transform_image_static(image, self.rotate_degrees,
                                           self.imagePadding)

    @staticmethod
    def transform_image_static(image, rotate_degrees, imagePadding):
        if image.shape[0] == 4:
            image = image[:3, :, :]
        # image = im_transform.rotate(image, self.rotate_degrees, resize=False)
        image = functional.rotate(image, rotate_degrees)
        image = np.pad(image, (
            (0, 0), (imagePadding, imagePadding),
            (imagePadding, imagePadding)),
                       "edge", )
        return torch.from_numpy(image)

    def getCellsFromImage(self, image,json_image_cells):
        indices = self.getCellIndices()
        cells = []
        for ((startX, endX), (startY, endY)), (x, y) in indices:
            if (not self.augmentTrain and not self.augmentTest):
                cell_indices =(transforms.functional.resize(
                    image[startY: endY, startX: endX].permute(
                        2, 0, 1),
                    [self.resizedCellWidth, self.resizedCellHeight]),
                              ((startX, endX), (startY, endY)), (x, y))
            else:
                cell_indices = (transforms.functional.resize(
                    image[startY: endY, startX: endX].permute(2, 0, 1),
                    [self.resultCellWidth, self.resultCellWidth])
                              if not self.augmentOriginal else image[
                                                               startY: endY,
                                                               startX: endX].permute(
                    2, 0, 1), ((startX, endX), (startY, endY)), (x, y))
            cell_id = f"cell_id_{str(uuid.uuid4())}"
            cells.append(
                {"cell_index":cell_indices[2],
                 "cell_id":cell_id,
                 "cell_image":cell_indices[0].tolist()}
            )
            json_image_cells[y][x]["cell_id"] = cell_id
        return cells

    def getCellIndices(self):
        indices = []
        for y in range(self.numCellsY):
            for x in range(self.numCellsX):
                startY = y * self.offsetY + self.centerOffsetY - self.cellHeight // 2
                startX = (
                        (0 if y % 2 == 0 else self.evenRowOffsetX)
                        + x * self.offsetX
                        + self.centerOffsetX
                        - self.cellWidth // 2
                )
                indices.append((((startX, startX + self.cellWidth),
                                 (startY, startY + self.cellHeight)), (x, y)))
        return indices



def main():
    jsonfile = import_from_json(filename="full_dataset_predictions.json")
    image_transformer = Imageprocessor(
        augmentTest=False,
        augmentTrain=False,
        augmentOriginal=False

    )
    json_file_extension_updated = replace_image_extension(jsonfile)

    folder_path = './Images'  # replace with your folder path
    available_images = os.listdir(folder_path)
    available_training_json = [frame for frame in json_file_extension_updated
                                 if
                                 frame[
                                     "filename"] in available_images]

    file_counter = 0
    for filename in os.listdir(folder_path):
        image = image_transformer.read_image(f"./Images/{filename}",
                                "png", save=None)
        for json_image_labels in available_training_json:
            if json_image_labels["filename"] == filename:
                cells = image_transformer.getCellsFromImage(image,json_image_labels["cells"])
                with open(f"./cells_image_data_{file_counter}.json", 'w') as f:
                    json.dump(cells, f)
                file_counter = file_counter + 1
    with open("./full_dataset_predictions_updated.json", 'w') as f:
        json.dump(available_training_json, f)



if __name__ == '__main__':
    main()
