import torch
import os
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from helper_functions import (import_from_json,export_to_json)
from Neural_Network_model import CellModel
# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
training_params_folder = '/content/SWP-Brood-Cell-AgeSoSe23-FU/Neural_Network/Gaussian_Blur/Modified_Model/Model_Parameters/'

def testing(testing_tensor:torch.tensor,true_ages):
    model = CellModel().to(device)
    criterion = nn.MSELoss()
    model.load_state_dict(torch.load(f'{training_params_folder}/parameters.pth'))
    data_mean = torch.load(f'{training_params_folder}/data_mean.pt')
    data_std = torch.load(f'{training_params_folder}/data_std.pt')

    true_ages_tensor = torch.tensor(true_ages).unsqueeze(1).to(device)

    # Normalize the test data using the mean and standard deviation of the training data
    test_data = testing_tensor.to(device)
    test_data = (test_data - data_mean) / data_std

    model.eval()

    # Disable gradient calculation to improve inference performance
    with torch.no_grad():
        # Pass the test data through the model
        predictions = model(test_data)

        loss_criterion = criterion(predictions, true_ages_tensor)
        loss = loss_criterion.item()

        return predictions,loss





if __name__ == "__main__":
    test_tensor_folder_path = '/content/SWP-Brood-Cell-AgeSoSe23-FU/Neural_Network/Gaussian_Blur/testing_tensor_data/tensors'
    test_true_ages_folder = '/content/SWP-Brood-Cell-AgeSoSe23-FU/Neural_Network/Gaussian_Blur/Predictions/True_test_labels'
    predictions_folder = '/content/SWP-Brood-Cell-AgeSoSe23-FU/Neural_Network/Gaussian_Blur/Predictions/Predicted_labels'
    for file_name in os.listdir(test_tensor_folder_path):
        testing_tensor = torch.load(
        f'{test_tensor_folder_path}/{file_name}')
        tensor_to_numpy = testing_tensor.numpy()
        tensor_image = tensor_to_numpy.transpose(0, 2, 3, 1)
        age_file_name = os.path.splitext(file_name)[0]
        true_ages = import_from_json(f'{test_true_ages_folder}/{age_file_name}.json')
        predictions,loss = testing(testing_tensor,true_ages)
        print(f"Loss in file {age_file_name}: {loss}")
        predictions_list = predictions.tolist()
        export_to_json(folder = predictions_folder,
                       filename = age_file_name,
                       file=predictions_list)

