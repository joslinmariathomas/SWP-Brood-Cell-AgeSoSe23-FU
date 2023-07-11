import os
import json
from helper_functions import import_from_json


class SplitTrainCells:
    imageWidth = 4992
    splitX = int(imageWidth * 0.4)
    splitX2 = int(imageWidth * 0.6)

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
        # with open("train_cells.json", 'w') as f:
        #     json.dump(self.train_cells, f)
        # with open("test_cells.json", 'w') as f:
        #     json.dump(self.test_cells, f)

    def get_train_test_cells(self,cell_image_data):
        cells_image_data_json = import_from_json(
            f"./cells_image_data/{cell_image_data}")
        cell_test_data = []
        cell_train_data = []
        cell_file_no_ext = os.path.splitext(cell_image_data)[0]
        for cell_data in cells_image_data_json:
            if self.addTrainOrTestIndex(
                    cell_data.get("cell_indices")
            ):
                cell_train_data.append(cell_data)
            else:
                cell_test_data.append(cell_data)
        with open(f"./training_data/{cell_file_no_ext}.json", 'w') as f:
            json.dump(cell_train_data, f)
        with open(f"./testing_data/{cell_file_no_ext}.json", 'w') as f:
            json.dump(cell_test_data, f)



def main():
    TrainTestSplit = SplitTrainCells()
    folder_path = './cells_image_data'
    TrainTestSplit.getTrainTestSplit(folder_path=folder_path)
    # TrainTestSplit.get_train_data()
    # TrainTestSplit.get_test_data()



if __name__ == '__main__':
    main()
