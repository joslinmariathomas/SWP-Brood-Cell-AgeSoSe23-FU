import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from helper_functions import (import_from_json,export_to_json)
from Neural_Network_model import CellModel
# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
training_params_folder = '/content/SWP-Brood-Cell-AgeSoSe23-FU/Neural_Network/Model_Parameters'

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
        print(loss)





if __name__ == "__main__":
    testing_tensor = torch.load(
        '/content/SWP-Brood-Cell-AgeSoSe23-FU/testing_tensor_data/tensors/scan_back_220810-044352-utc_test.pt')
    true_ages = import_from_json(
        '/content/SWP-Brood-Cell-AgeSoSe23-FU/Predictions/True_test_labels/scan_back_220810-044352-utc_test.json')
    testing(testing_tensor,true_ages)
