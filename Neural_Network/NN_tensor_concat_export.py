import torch
import os
from helper_functions import (import_from_json, export_to_json)

full_age_data = import_from_json('../full_dataset_with_ages_with_ids.json')


def get_cell_ids(file_name, cell_id_list,
                 cell_age_list, filename_list,
                 train_or_test, labels_id_folder):
    label_file_name = os.path.splitext(file_name)[0]
    cell_ids = import_from_json(f'{labels_id_folder}/{label_file_name}.json')
    filename_updated = label_file_name.replace(f"_{train_or_test}", "")
    filenames = [filename_updated] * len(cell_ids)
    cells_data = get_cells_data(filename=filename_updated,
                                age_data=full_age_data,
                                train_or_test=train_or_test)
    for cell_id in cell_ids:
        age = get_age(cells_data=cells_data,
                      cell_identity=cell_id)
        cell_age_list.append(age)
    cell_id_list.extend(cell_ids)
    filename_list.extend(filenames)
    return cell_id_list, cell_age_list, filename_list


def get_age(cells_data, cell_identity):
    for row_cell in cells_data:
        for column_cell in row_cell:
            cell_id = column_cell.get("cell_id")
            if cell_id == cell_identity:
                age = column_cell.get("age")
                return age


def get_cells_data(age_data, filename, train_or_test):
    for image in age_data:
        image_name = os.path.splitext(image.get("filename"))[0]
        if image_name == filename:
            cells_data = image.get("cells")
            return cells_data


def save_tensor_and_ages(train_or_test: str, tensor_folder_path,
                         labels_folder, folder_to_save):
    cell_id_list = []
    cell_age_list = []
    model_list = []
    filename_list = []
    for file_name in os.listdir(tensor_folder_path):
        if train_or_test == "test":
            cell_age_list = []
        cell_id_list, cell_age_list, filename_list = get_cell_ids(
            file_name=file_name, cell_id_list=cell_id_list,
            cell_age_list=cell_age_list,
            train_or_test=train_or_test,
            labels_id_folder=labels_folder,
            filename_list=filename_list)
        if file_name.endswith('.pt'):
            file_path = os.path.join(tensor_folder_path, file_name)
            model = torch.load(file_path)
            model_list.append(model)
        if train_or_test =="test":
            label_file_name = os.path.splitext(file_name)[0]
            pred_label_folder = '../Predictions/True_test_labels'
            export_to_json(filename=f"{label_file_name}",
                           folder=pred_label_folder,
                           file=cell_age_list)


    stacked_model = torch.cat(model_list, dim=0)
    if train_or_test =="train":

        export_to_json(filename=f"age_for_tensors_{train_or_test}",
                       folder=folder_to_save,
                       file=cell_age_list)
    export_to_json(filename=f"cell_ids_{train_or_test}",
                   folder=folder_to_save,
                   file=cell_id_list)
    export_to_json(filename=f"cell_filenames_{train_or_test}",
                   folder=folder_to_save,
                   file=filename_list)
    torch.save(stacked_model,
               f'{folder_to_save}/{train_or_test}_tensor.pt')


def main():
    training_data_folder = '../training_tensor_data/tensors/'
    training_labels_folder = '../training_tensor_data/labels/'
    folder_to_save = './'
    # save_tensor_and_ages(train_or_test="train",
    #                      tensor_folder_path=training_data_folder,
    #                      labels_folder=training_labels_folder,
    #                      folder_to_save=folder_to_save)
    testing_data_folder = '../testing_tensor_data/tensors/'
    testing_labels_folder = '../testing_tensor_data/labels/'
    folder_to_save = './'
    save_tensor_and_ages(train_or_test="test",
                         tensor_folder_path=testing_data_folder,
                         labels_folder=testing_labels_folder,
                         folder_to_save=folder_to_save)


if __name__ == "__main__":
    main()
