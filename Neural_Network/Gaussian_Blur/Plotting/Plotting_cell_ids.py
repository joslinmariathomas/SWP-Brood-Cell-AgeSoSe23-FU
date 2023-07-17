import os
import torch
from helper_functions import import_from_json
import matplotlib.dates as mdates
from datetime import datetime,timedelta
import torch
import os
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from helper_functions import (import_from_json,export_to_json)
from Neural_Network.Image_Augmentation.Neural_Network_model_ImAug import CellModel
# Check for GPU availability
training_params_folder = '/content/SWP-Brood-Cell-AgeSoSe23-FU/Neural_Network/Gaussian_Blur/Modified_Model/Model_Parameters'

row_id = 6
col_id = 4


full_age_data = import_from_json(
    '/content/SWP-Brood-Cell-AgeSoSe23-FU/full_dataset_with_ages_with_ids.json')
training_folder = "/content/SWP-Brood-Cell-AgeSoSe23-FU/Neural_Network/Gaussian_Blur/training_tensor_data/tensors"
testing_folder = "/content/SWP-Brood-Cell-AgeSoSe23-FU/Neural_Network/Gaussian_Blur/testing_tensor_data/tensors"
file_name_cell_id = {}

for image in full_age_data:
    filename = image.get("filename")
    cells = image.get("cells")
    cell_content = cells[row_id][col_id]
    cell_content["timestamp"] = image.get("time")
    if cell_content.get("cell_id") is not None:
        file_name_cell_id[filename] = cell_content




#
def testing(testing_tensor:torch.tensor,true_ages):
    model = CellModel()
    criterion = nn.MSELoss()
    checkpoint = torch.load(f'{training_params_folder}/parameters.pth',
                            map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint)
    data_mean = torch.load(f'{training_params_folder}/data_mean.pt',
                           map_location=torch.device('cpu'))
    data_std = torch.load(f'{training_params_folder}/data_std.pt',
                          map_location=torch.device('cpu'))

    true_ages_tensor = torch.tensor(true_ages).unsqueeze(1)

    # Normalize the test data using the mean and standard deviation of the training data
    test_data = testing_tensor
    test_data = (test_data - data_mean) / data_std

    model.eval()

    # Disable gradient calculation to improve inference performance
    with torch.no_grad():
        # Pass the test data through the model
        predictions = model(test_data)
        predictions = torch.relu(predictions)
        loss_criterion = criterion(predictions, true_ages_tensor)
        loss = loss_criterion.item()

        return predictions,loss



# check training data or testing

def train_or_test():
    filename = "scan_back_220810-044352-utc.png"
    cell_id = file_name_cell_id.get(filename)["cell_id"]

    train_cell_ids = import_from_json(
        "/content/SWP-Brood-Cell-AgeSoSe23-FU/training_tensor_data/labels/scan_back_220810-044352-utc.json")
    test_cell_ids = import_from_json(
        "/content/SWP-Brood-Cell-AgeSoSe23-FU/testing_tensor_data/labels/scan_back_220810-044352-utc.json")

    if cell_id in train_cell_ids:
        index = train_cell_ids.index(cell_id)
        return "train", index
    else:
        index = test_cell_ids.index(cell_id)
        return "test", index


def get_predictions(folder):
    tensor_list = []
    true_ages = []
    time = []
    for file in file_name_cell_id:
        file_name_no_ext = os.path.splitext(file)[0]
        tensor = torch.load(f'{folder}/{file_name_no_ext}.pt')
        age = file_name_cell_id.get(file).get("age")
        timestamp = file_name_cell_id.get(file).get("timestamp")
        true_ages.append(age)
        time.append(timestamp)
        tensor_list.append(tensor[index])
    tensor_combined = torch.stack(tensor_list, dim=0)
    tensor_combined = tensor_combined
    predictions,loss = testing(testing_tensor=tensor_combined,
                               true_ages=true_ages)
    return predictions,loss,true_ages,time
if __name__ == "__main__":
    train_test, index = train_or_test()

    if train_test == "train":
        folder = training_folder
    else:
        folder = testing_folder
    prediction,loss,true_ages,time = get_predictions(folder)
    converted_times = []
    for timestamp in time:
        dt = datetime.fromtimestamp(timestamp)
        converted_times.append(dt.strftime("%Y-%m-%d"))


    prediction= prediction.tolist()
    prediction_ages = [(x[0]) for x in prediction]
    converted_times = [datetime.fromtimestamp(timestamp) for timestamp in time]

    # Plotting
    plt.plot(converted_times, prediction_ages, label='Prediction Ages')
    plt.plot(converted_times, true_ages, label='True Ages')

    # Customize plot
    plt.xlabel('Time')
    plt.ylabel('Ages')
    plt.title(f'Age Prediction Over Time for cell {row_id},{col_id}')
    plt.legend()
    ticks = []
    tick_labels = []
    current_date = converted_times[0].date()
    while current_date <= converted_times[-1].date():
        ticks.append(current_date)
        tick_labels.append(current_date.strftime('%Y-%m-%d'))
        current_date += timedelta(days=5)

    plt.xticks(ticks, tick_labels,
               rotation=22)  # Rotate tick labels by 45 degrees
    plt.axvspan(19222.23611111111, 19226.711203703704, color='grey', alpha=0.5)
    plt.axvspan(19237.36111111111, 19243.708333333332, color='grey', alpha=0.5)
    # Display the plot
    plt.show()



