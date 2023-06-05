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

        if boundaries[0][1] < self.splitX or boundaries[0][1] >= self.splitX2:
            return True
        else:
            return False
    def getTrainTestSplit(self,folder_path):
        available_cells = os.listdir(folder_path)
        for cell_image_data in available_cells:
            self.get_train_test_cells(
                cell_image_data=cell_image_data)
        with open("train_cells.json", 'w') as f:
            json.dump(self.train_cells, f)
        with open("test_cells.json", 'w') as f:
            json.dump(self.test_cells, f)

    def get_train_test_cells(self,cell_image_data):
        cells_image_data_json = import_from_json(
            f"./cells_image_data/{cell_image_data}")
        for cell_data in cells_image_data_json:
            if self.addTrainOrTestIndex(
                    cell_data.get("cell_indices")
            ):
                cell_data["Train_Test"] = "Train"
                cell_dict = {"filename":cell_image_data,"cell_id":cell_data["cell_id"]}
                self.train_cells.append(cell_dict)
            else:
                cell_dict = {"filename": cell_image_data,
                             "cell_id": cell_data["cell_id"]}
                self.test_cells.append(cell_dict)

    def get_train_data(self):
        grouped_data = self.get_cell_for_filename(self.train_cells)
        for filename in grouped_data:
            trainining_cells = []
            for cell_id in grouped_data[filename]:
                cell_train_json =  import_from_json(
                f"./cells_image_data/{filename}")

                for cells in cell_train_json:
                    if cell_id == cells.get("cell_id"):
                        trainining_cells.append(cells)
                        break
            filename_train = f"{os.path.splitext(filename)[0]}_train"
            with open(f"./training_data/{filename_train}.json", 'w') as f:
                json.dump(trainining_cells, f)

    def get_test_data(self):
        grouped_data = self.get_cell_for_filename(self.test_cells)
        for filename in grouped_data:
            testing_cells = []
            for cell_id in grouped_data[filename]:
                cell_test_json = import_from_json(
                    f"./cells_image_data/{filename}")

                for cells in cell_test_json:
                    if cell_id == cells.get("cell_id"):
                        testing_cells.append(cells)
                        break
            filename_test = f"{os.path.splitext(filename)[0]}_test"
            with open(f"./testing_data/{filename_test}.json", 'w') as f:
                json.dump(testing_cells, f)

    def get_cell_for_filename(self, cells_dict):
        grouped_data = {}
        for data_dict in cells_dict:
            filename = data_dict["filename"]
            data = data_dict["cell_id"]
            if filename in grouped_data:
                grouped_data[filename].append(data)
            else:
                grouped_data[filename] = [data]
        return grouped_data


def main():
    TrainTestSplit = SplitTrainCells()
    folder_path = './cells_image_data'
    TrainTestSplit.getTrainTestSplit(folder_path=folder_path)
    TrainTestSplit.get_train_data()
    TrainTestSplit.get_test_data()



if __name__ == '__main__':
    main()
