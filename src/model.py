# import module
import torch
import torch.nn as nn
import torch.nn.functional as F

# model definition
class TakeoffClassModel(nn.Module):
    def __init__(self):
        super(TakeoffClassModel, self).__init__()        
        self.conv1 = nn.Conv2d(3, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.fc = nn.Sequential(
            nn.Linear(451584, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 3)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x