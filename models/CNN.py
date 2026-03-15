import torch.nn as nn

class CNNModelScratch(nn.Module):
    def __init__(self):
        super().__init__()
        ## Convolutional layers
        self.conv1d = nn.Conv2d(in_channels=1, out_channels=3, kernel_size=4, stride=1) 
        ## Pooling layers
        self.maxpool1d = nn.MaxPool2d(kernel_size=4, stride=1)
        ## Fully Connected layers
        self.flatten = nn.Flatten()
        self.fc = nn.Sequential(
            nn.Linear(1452, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
            nn.ReLU()
        )
        

    def forward(self, x):
        x = self.conv1d(x)
        x = self.maxpool1d(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x
