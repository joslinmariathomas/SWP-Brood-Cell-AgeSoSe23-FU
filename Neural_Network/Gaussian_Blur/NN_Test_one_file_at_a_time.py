import torch
import torchvision
import os
import random

import matplotlib.pyplot as plt
from helper_functions import (import_from_json,export_to_json)
from Neural_Network.Gaussian_Blur.Modified_Model.Neural_Network_model import CellModel
# Check for GPU availability
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# training_params_folder = '/content/SWP-Brood-Cell-AgeSoSe23-FU/Neural_Network/Model_Parameters'


if __name__ == "__main__":
    test_tensor_folder_path = '../testing_tensor_data/tensors'
    test_true_ages_folder = '../Predictions/True_test_labels'
    files = (os.listdir(test_tensor_folder_path))
    i = random.randint(0, 115)
    file_name = files[i]
    file_no_ext = os.path.splitext(file_name)[0]

    predictions =  import_from_json(f"../Predictions/Predicted_labels/{file_no_ext}.json")
    predictions = [round(pred[0],2) for pred in predictions]
    annotations = import_from_json(f"../Predictions/True_test_labels/{file_no_ext}.json")
    annotations = [round(age,2) for age in annotations]

    testing_tensor = torch.load(
        f'{test_tensor_folder_path}/{file_name}')
    padding = 30
    grid_image = torchvision.utils.make_grid(testing_tensor, nrow=8
                                             , padding=padding)

    # Convert the grid image tensor to a numpy array
    grid_image_np = grid_image.permute(1, 2, 0).numpy()

    # Display the grid of images
    plt.figure(figsize=(12, 11))
    plt.imshow(grid_image_np)
    plt.axis('off')

    for i, (annotation, image, prediction) in enumerate(
            zip(annotations, testing_tensor, predictions)):
        annotation_text = f"{annotation}\n {prediction}"
        img_width = image.shape[2]
        y_offset = 50 + ((i // 8) * 5)
        annotation_color = 'green'
        prediction_color = 'red'
        plt.text((i % 8) * (img_width + padding), (i // 8) * 90 + y_offset,
                  f"{annotation}", color=annotation_color, ha='left', va='top')
        plt.text((i % 8) * (img_width + padding),
                 (i // 8) * 90 + y_offset + 20,
                 f"{prediction}", color=prediction_color, ha='left', va='top')

    plt.show()



