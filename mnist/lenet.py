import torch
from torch import nn
from torch.nn.modules.activation import ReLU


class LeNet(nn.Module):
    features: nn.Sequential
    classifier: nn.Sequential

    def __init__(self, num_classes: int):
        super(LeNet, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        '''
        self.features = nn.Sequential(            
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2),
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2),
            nn.Conv2d(in_channels=16, out_channels=120, kernel_size=5, stride=1),
            nn.Tanh()
        )'''
        '''
        self.classifier = nn.Sequential(
            nn.Linear(in_features=120, out_features=84),
            nn.Tanh(),
            nn.Linear(in_features=84, out_features=num_classes),
        )'''

        self.classifier = nn.Sequential(
            nn.Linear(32*5*5, 120),
            nn.ReLU(),
            nn.Linear(120, 10)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = torch.flatten(x, 1)
        #x = x.view(x.size()[0], -1)
        x = self.classifier(x)
        return x