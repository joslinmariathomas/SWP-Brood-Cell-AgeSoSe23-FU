import torch
import random
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from helper_functions import (import_from_json,export_to_json)
from Neural_Network_model import CellModel
from Image_Augmentation import augment_image
# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
random.seed(1)
torch.manual_seed(1)
model = CellModel().to(device)

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.001)

# Load the actual outputs saved in a list
actual_outputs = import_from_json('/content/SWP-Brood-Cell-AgeSoSe23-FU/Neural_Network/Image_Augmentation/age_for_tensors_train.json')  # Replace with your actual output values

# Load the .pt tensor file
tensor_data = torch.load('/content/SWP-Brood-Cell-AgeSoSe23-FU/Neural_Network/Image_Augmentation/train_tensor.pt')

# Convert tensor_data into a list of tensors
tensor_list = list(tensor_data)

# Stack the tensors in the list and mov\e them to the GPU
batch_tensor = torch.stack(tensor_list).to(device)

# Prepare the target tensor and move to GPU (if applicable)
target_tensor = torch.tensor(actual_outputs).unsqueeze(1).to(device)


# Split data into training and validation sets
validation_ratio = 0.2
num_samples = len(tensor_data)
num_validation = int(num_samples * validation_ratio)
num_training = num_samples - num_validation

training_data = tensor_data[:num_training]
validation_data = tensor_data[num_training:]

training_targets = target_tensor[:num_training]
validation_targets = target_tensor[num_training:]

training_data = training_data.to(device)
validation_data = validation_data.to(device)
training_targets = training_targets.to(device)
validation_targets = validation_targets.to(device)

# Compute mean and standard deviation of the training data
data_mean = torch.mean(training_data, dim=0)
data_std = torch.std(training_data, dim=0)

# Normalize the training data
training_data = (training_data - data_mean) / data_std

# Normalize the validation data using the mean and standard deviation of the training data
validation_data = (validation_data - data_mean) / data_std

# Perform training iterations
num_epochs = 100
batch_size = 32  # Choose an appropriate batch size
num_training_batches = (num_training - 1) // batch_size + 1
num_validation_batches = (num_validation - 1) // batch_size + 1

# Lists to store losses for plotting
train_losses = []
val_losses = []

for epoch in range(num_epochs):
    running_loss = 0.0

    # Shuffle the data and target tensor together for each epoch
    indices = torch.randperm(num_training)
    shuffled_data = training_data[indices]
    shuffled_targets = training_targets[indices]

    model.train()
    for batch_idx in range(num_training_batches):
        # Get the current batch
        start_idx = batch_idx * batch_size
        end_idx = start_idx + batch_size

        batch_target = shuffled_targets[start_idx:end_idx]
        batch_input_list = []

        for i in range(start_idx, end_idx):
            image_tensor = shuffled_data[i]
            augmented_tensor = augment_image(image_tensor)
            batch_input_list.append(augmented_tensor)
        batch_input = torch.stack(batch_input_list,dim=0)
        batch_input = batch_input.to(device)
        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        output = model(batch_input)

        # Calculate the loss
        loss = criterion(output, batch_target)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    # Print the average loss for the epoch
    avg_train_loss = running_loss / num_training_batches
    train_losses.append(avg_train_loss)

    # Validation phase
    model.eval()
    with torch.no_grad():
        val_loss = 0.0
        for batch_idx in range(num_validation_batches):
            start_idx = batch_idx * batch_size
            end_idx = start_idx + batch_size
            batch_input = validation_data[start_idx:end_idx]
            batch_target = validation_targets[start_idx:end_idx]

            output = model(batch_input)
            loss = criterion(output, batch_target)
            val_loss += loss.item()

        avg_val_loss = val_loss / num_validation_batches
        val_losses.append(avg_val_loss)
    print(
        f"Epoch [{epoch + 1}/{num_epochs}], Training Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}")


# Plot the loss graph
plt.plot(range(1, len(train_losses) + 1), train_losses, label='Training Loss')
plt.plot(range(1, len(val_losses) + 1), val_losses, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()

