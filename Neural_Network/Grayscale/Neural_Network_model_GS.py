import torch
import torch.nn as nn

class CellModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=2, padding=1),
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
            nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=0),
            nn.ReLU()
        )
        for m in self.layers.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)

        # Add dropout layers
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        x = self.layers(x)
        x = torch.flatten(x, start_dim=1)
        x = self.dropout(x)  # Apply dropout
        return x
