import torch
# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the model architecture
import torch
import torch.nn as nn

class CellModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=5, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=5, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
              # Output layer with 1 channel
        )
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(64 * 3 * 3, 1)
    def forward(self, x):
        x = self.layers(x)
        x = self.flatten(x)
        x = self.linear(x) # flatten all dimensions except batch
        return x

