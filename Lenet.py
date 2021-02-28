import torch.nn as nn
import torch
import torch.nn.functional as F 

class LeNet(nn.Module):
    def __init__(self, output_dim):
        super().__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(3, 3)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )
        
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=(3, 3)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=(3, 3)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
    
        self.layer4 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=(3, 3)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.fc1 = nn.Linear(in_features=64*5*5, out_features=120)
        self.fc2 = nn.Linear(in_features=120, out_features=84)
        self.fc3 = nn.Linear(in_features=84, out_features=10)

    def forward(self, x):
      out = self.layer1(x)
      out = self.layer2(out)
      out = self.layer3(out)
      out = self.layer4(out)
      out = out.view(out.size(0), -1)
      out = F.dropout(F.relu(self.fc1(out)), 0.1)
      out = F.dropout(F.relu(self.fc2(out)), 0.1)
      out = self.fc3(out)
      return F.log_softmax(out)
