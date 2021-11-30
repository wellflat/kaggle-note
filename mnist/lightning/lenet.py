from typing import Tuple
import torch
from torch import nn, optim, Tensor
from torch.optim.lr_scheduler import StepLR
from pytorch_lightning import LightningModule


class LeNet(LightningModule):
    features: nn.Sequential
    classifier: nn.Sequential
    criterion: nn.CrossEntropyLoss

    def __init__(self, num_classes: int):
        super(LightningModule, self).__init__()
        
        self.features = nn.Sequential(            
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2),
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2),
            nn.Conv2d(in_channels=16, out_channels=120, kernel_size=5, stride=1),
            nn.Tanh()
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(in_features=120, out_features=84),
            nn.Tanh(),
            nn.Linear(in_features=84, out_features=num_classes),
        )

        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x: Tensor) -> Tensor:
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=0.01)
        scheduler = StepLR(optimizer, step_size=25, gamma=0.1)
        return (optimizer, scheduler)

    def training_step(self, batch:Tuple[Tensor, Tensor], batch_nb:int) -> Tensor:
        inputs, targets = batch
        outputs = self.forward(inputs)
        loss: Tensor = self.criterion(outputs, targets)
        return loss

    def validation_step(self, batch:Tuple[Tensor, Tensor], batch_nb:int):
        inputs, targets = batch
        outputs = self.net(inputs)
        loss: Tensor = self.criterion(outputs, targets)
    
    def training_epoch_end(self, outputs: EPOCH_OUTPUT) -> None:
        return super().training_epoch_end(outputs)

    def validation_epoch_end(self, outputs: EPOCH_OUTPUT) -> None:
        return super().validation_epoch_end(outputs)





    