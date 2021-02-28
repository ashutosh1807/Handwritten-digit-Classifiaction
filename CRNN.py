import torch.nn as nn
import torch
import torch.nn.functional as F 


class CRNN(nn.Module):
    
    def __init__(self, cnn_output_height, gru_hidden_size, gru_num_layers, num_classes):
        super(CRNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(3, 3)),
            nn.InstanceNorm2d(32),
            nn.ReLU(),
        )
        
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=(3, 3)),
            nn.InstanceNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=(3, 3)),
            nn.InstanceNorm2d(64),
            nn.ReLU(),
        )
    
        self.layer4 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=(3, 3)),
            nn.InstanceNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.gru_input_size = cnn_output_height * 64
        self.gru = nn.GRU(self.gru_input_size, gru_hidden_size, gru_num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(gru_hidden_size * 2, num_classes)

    def forward(self, x):
        batch_size = x.shape[0]
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = out.permute(0, 3, 2, 1)
        out = out.reshape(batch_size, -1, self.gru_input_size)
        out, _ = self.gru(out)
        out = torch.stack([F.log_softmax(self.fc(out[i]), dim=-1) for i in range(out.shape[0])])
        return out
