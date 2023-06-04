import os
import json
from helper_functions import import_from_json


class SplitTrainCells:
    imageWidth = 4992
    splitX = int(imageWidth * 0.4)
    splitX2 = int(imageWidth * 0.6)

    def __init__(self):
        self.train_cells = []
        self.test_cells = []

    def addTrainOrTestIndex(self, boundaries):
        # if boundaries[0][1] < self.splitX:
        if boundaries[0][1] < self.splitX or boundaries[0][1] >= self.splitX2:
            return True
        else:
            return False
    def get_train_test_cells(self,cell_image_data):
        cells_image_data = import_from_json(
            f"./cells_image_data/{cell_image_data}")
        for cell_data in cells_image_data:
            if self.addTrainOrTestIndex(
                    cell_data.get("cell_indices")
            ):
                cell_data["Train_Test"] = "Train"
                self.train_cells.append(cell_data["cell_id"])
            else:
                self.test_cells.append(cell_data["cell_id"])





def main():
    TrainTestSplit = SplitTrainCells()

    folder_path = './cells_image_data'
    available_cells = os.listdir(folder_path)

    for cell_image_data in available_cells:
        TrainTestSplit.get_train_test_cells(cell_image_data= cell_image_data)

    with open("./train_cells.json", 'w') as f:
        json.dump(TrainTestSplit.train_cells, f)
    with open("./test_cells.json", 'w') as f:
        json.dump(TrainTestSplit.test_cells, f)

if __name__ == '__main__':
    main()
