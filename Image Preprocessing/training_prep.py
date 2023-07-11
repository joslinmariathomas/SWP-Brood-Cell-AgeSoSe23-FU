import torch
import os
import json

from helper_functions import import_from_json


def convert_training_data_to_tensor():
    training_data = os.listdir("./training_data/")
    for data in training_data:
        file_data = import_from_json(f"./training_data/{data}")
        image_tensor_list = []
        cell_id_in_order = []
        for cell_details in file_data:
            cell_id_in_order.append(cell_details["cell_id"])
            image_tensor_list.append(torch.tensor(cell_details["cell_image"]))
        filename = f"{os.path.splitext(data)[0]}"
        with open(f"./training_tensor_data/labels/{filename}.json", 'w') as f:
            json.dump(cell_id_in_order, f)
        combined_tensor = torch.stack(image_tensor_list)
        save_path = f"./training_tensor_data/tensors/{filename}.pt"
        torch.save(combined_tensor, save_path)


def convert_testing_data_to_tensor():
    testing_data = os.listdir("./testing_data/")
    for data in testing_data:
        file_data = import_from_json(f"./testing_data/{data}")
        image_tensor_list = []
        cell_id_in_order = []
        for cell_details in file_data:
            cell_id_in_order.append(cell_details["cell_id"])
            image_tensor_list.append(torch.tensor(cell_details["cell_image"]))
        filename = f"{os.path.splitext(data)[0]}"
        with open(f"./testing_tensor_data/labels/{filename}.json", 'w') as f:
            json.dump(cell_id_in_order, f)
        combined_tensor = torch.stack(image_tensor_list)
        save_path = f"./testing_tensor_data/tensors/{filename}.pt"
        torch.save(combined_tensor, save_path)

if __name__ == '__main__':
    convert_training_data_to_tensor()
    convert_testing_data_to_tensor()


